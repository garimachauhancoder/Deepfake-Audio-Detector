"""
Features Module for Multi-Branch Speech Deepfake Detection System
Extracts the exact 2,370-dimensional multi-view representation:
  - Acoustic View (830 dims): 62 handcrafted (librosa) + 768 HuBERT Layer 2
  - Phonetic View (769 dims): 768 HuBERT Layer 6 + 1 transition smoothness
  - Semantic View (768 dims): 768 HuBERT Layer 12
  - Cross-View Consistency (3 dims): Pairwise cosine similarities
"""

import numpy as np
import librosa
import torch


def get_device():
    """Detect available accelerator (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_hubert_models(model_name="facebook/hubert-base-ls960", device=None):
    """
    Loads HuggingFace AutoFeatureExtractor and HubertModel with output_hidden_states=True.
    Safe to use with @st.cache_resource in Streamlit.
    """
    from transformers import AutoFeatureExtractor, HubertModel
    
    if device is None:
        device = get_device()
        
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    return extractor, model, device


def extract_handcrafted_acoustic(wav, sr=16000):
    """
    Extracts 62-dimensional handcrafted acoustic features using librosa:
      - 20 MFCC means
      - 20 MFCC standard deviations
      - 20 Delta-MFCC means
      - 1 Spectral Centroid mean
      - 1 RMS Energy standard deviation
    """
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)
    rms = librosa.feature.rms(y=wav)
    
    acoustic_handcrafted = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        centroid.mean(axis=1),
        rms.std(axis=1)
    ]).astype(np.float32)
    
    return acoustic_handcrafted


def extract_all_three_features(wav, extractor, hubert_model, device="cpu", sr=16000):
    """
    Extracts the full 2,370-dimensional feature vector exactly matching the
    ASVspoof 2021 LA training pipeline:
      [0:830]     -> Acoustic View (62 handcrafted + 768 HuBERT L2)
      [830:1599]  -> Phonetic View (768 HuBERT L6 + 1 transition smoothness)
      [1599:2367] -> Semantic View (768 HuBERT L12)
      [2367:2370] -> Cross-View Consistency (3 cosine similarities)
      
    Returns:
        fused_vector (np.ndarray): Shape (2370,) float32
        forensic_dict (dict): Detailed intermediate representations for visualizations
    """
    # 1. ACOUSTIC Handcrafted (62 dims)
    acoustic_handcrafted = extract_handcrafted_acoustic(wav, sr=sr)
    
    # 2. HuBERT Multi-Layer Hidden States
    inputs = extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = hubert_model(**inputs)
        hidden = outputs.hidden_states  # Tuple of 13 tensors (Layer 0 to 12), shape: (1, T, 768)

        # Layer 2: Acoustic SSL representation (Early layer)
        h2 = hidden[2]  # (1, T, 768)
        acoustic_ssl = h2.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

        # Layer 6: Phonetic representation (Middle layer)
        h6 = hidden[6]  # (1, T, 768)
        phonetic_ssl = h6.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        
        # Frame-to-frame L2 norm differences (transition flux across time)
        # diff shape: (1, T-1, 768)
        diff_l2 = torch.norm(h6[:, 1:, :] - h6[:, :-1, :], dim=-1)  # (1, T-1)
        transition_smoothness = float(diff_l2.std().item())
        frame_transition_flux = diff_l2.squeeze(0).cpu().numpy()  # (T-1,)

        # Layer 12: Semantic & Contextual Identity (Final layer)
        h12 = hidden[12]  # (1, T, 768)
        semantic_ssl = h12.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        
        # Frame-by-frame cosine harmony between Layer 2 (Acoustic) and Layer 6 (Phonetic)
        # h2: (1, T, 768), h6: (1, T, 768)
        h2_norm = h2 / (torch.norm(h2, dim=-1, keepdim=True) + 1e-8)
        h6_norm = h6 / (torch.norm(h6, dim=-1, keepdim=True) + 1e-8)
        frame_harmony = (h2_norm * h6_norm).sum(dim=-1).squeeze(0).cpu().numpy()  # (T,)

    # 3. Cross-View Consistency: Pairwise cosine similarities
    norm_ac = np.linalg.norm(acoustic_ssl) + 1e-8
    norm_ph = np.linalg.norm(phonetic_ssl) + 1e-8
    norm_sem = np.linalg.norm(semantic_ssl) + 1e-8

    cos_ac_ph = float(np.dot(acoustic_ssl, phonetic_ssl) / (norm_ac * norm_ph))
    cos_ph_sem = float(np.dot(phonetic_ssl, semantic_ssl) / (norm_ph * norm_sem))
    cos_ac_sem = float(np.dot(acoustic_ssl, semantic_ssl) / (norm_ac * norm_sem))

    # Assemble complete 2,370-dim fused vector
    fused_vector = np.concatenate([
        acoustic_handcrafted,           # [0:62]
        acoustic_ssl,                   # [62:830]    -> Total Acoustic: 830
        phonetic_ssl,                   # [830:1598]
        np.array([transition_smoothness], dtype=np.float32),  # [1598:1599] -> Total Phonetic: 769
        semantic_ssl,                   # [1599:2367] -> Total Semantic: 768
        np.array([cos_ac_ph, cos_ph_sem, cos_ac_sem], dtype=np.float32)  # [2367:2370] -> Consistency: 3
    ]).astype(np.float32)

    assert fused_vector.shape[0] == 2370, f"Expected 2370 features, got {fused_vector.shape[0]}"

    # Time axis for frames (HuBERT 20ms frame stride = 50 frames/sec approx)
    n_frames = len(frame_harmony)
    time_frames = np.linspace(0, 4.0, n_frames, endpoint=False)
    time_flux = np.linspace(0, 4.0, len(frame_transition_flux), endpoint=False)

    # Detect peak flux timestamp for forensic anomaly/splice marker
    if len(frame_transition_flux) > 0:
        peak_idx = int(np.argmax(frame_transition_flux))
        peak_flux_time = float(time_flux[peak_idx])
    else:
        peak_flux_time = 2.84

    forensic_dict = {
        "acoustic_handcrafted": acoustic_handcrafted,
        "acoustic_ssl": acoustic_ssl,
        "phonetic_ssl": phonetic_ssl,
        "transition_smoothness": transition_smoothness,
        "semantic_ssl": semantic_ssl,
        "cos_ac_ph": cos_ac_ph,
        "cos_ph_sem": cos_ph_sem,
        "cos_ac_sem": cos_ac_sem,
        "frame_harmony": frame_harmony,
        "frame_transition_flux": frame_transition_flux,
        "time_frames": time_frames,
        "peak_flux_time": peak_flux_time
    }

    return fused_vector, forensic_dict

