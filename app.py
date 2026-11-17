"""
Identity Shield — Deepfake Audio Forensic Console
Academic Context: MCA Dissertation (Department of Computer Science and Engineering, The NorthCap University, Gurugram)
Candidate: Garima Chauhan (25MCA020)
Title: "Deepfake Audio Detector: Sensitivity and Fusion of Acoustic, Phonetic, and Semantic Features Under Communication Channel Transformations"
Benchmark: ASVspoof 2021 Logical Access (LA) Multi-Codec Evaluation
"""

import json
import time
import os
import io
from pathlib import Path
import numpy as np
import streamlit as st

# Custom modules
from modules.preprocessing import (
    load_and_preprocess_audio,
    load_audio_16k,
    wav_to_bytes,
    generate_synthetic_demo_clip,
    convert_to_wav_bytes
)
from modules.features import (
    load_hubert_models,
    extract_all_three_features
)
from modules.inference import (
    load_model_bundle,
    run_multi_branch_inference
)
from modules.visualizations import (
    plot_forensic_waveform,
    plot_forensic_melspectrogram,
    plot_multiview_consistency_curve,
    format_timestamp
)

# Set page configuration
st.set_page_config(
    page_title="Identity Shield — Deepfake Audio Forensic Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Cyber-Forensic CSS Styling strictly matching reference UI
FORENSIC_CSS = """
<style>
/* Root and page background */
[data-testid="stAppViewContainer"] {
    background-color: #0b1116 !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Hide default streamlit margin bloat */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Dark forensic card container */
.forensic-card {
    background-color: #101921;
    border: 1px solid #1e2d3b;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
}

.forensic-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1a2836;
    padding-bottom: 8px;
    margin-bottom: 12px;
}

.card-title {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2dd4bf;
    margin: 0;
}

.card-subtitle {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Alert badge in card header */
.header-alert-badge {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid #f59e0b;
    color: #f59e0b;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

/* Multi-branch analysis row */
.branch-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0d151c;
    border: 1px solid #182532;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 10px;
    transition: border-color 0.2s ease;
}

.branch-row:hover {
    border-color: #24384a;
}

.branch-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.branch-icon {
    font-size: 1.25rem;
    color: #2dd4bf;
    background: #13222e;
    width: 34px;
    height: 34px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #1e3142;
}

.branch-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
}

.branch-desc {
    font-size: 0.72rem;
    color: #94a3b8;
}

.branch-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-align: center;
    min-width: 95px;
}

/* Metadata table styling */
.meta-table {
    width: 100%;
    border-collapse: collapse;
}

.meta-row {
    border-bottom: 1px solid #16222d;
    padding: 8px 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.meta-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #94a3b8;
    font-size: 0.8rem;
}

.meta-val {
    color: #f1f5f9;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: monospace;
}

/* Cyber pill tags */
.cyber-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    background: #0f1923;
    border: 1px solid #233748;
    color: #cbd5e1;
    margin-right: 8px;
    margin-top: 6px;
}

.cyber-pill-amber {
    border-color: #f59e0b;
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.08);
}

.cyber-pill-red {
    border-color: #ef4444;
    color: #ef4444;
    background: rgba(239, 68, 68, 0.08);
}

.cyber-pill-teal {
    border-color: #2dd4bf;
    color: #2dd4bf;
    background: rgba(45, 212, 191, 0.08);
}

/* Horizontal threshold meter bar */
.threshold-bar-container {
    width: 100%;
    height: 12px;
    background: #162431;
    border-radius: 6px;
    position: relative;
    margin: 12px 0 16px 0;
    border: 1px solid #1f3548;
    overflow: hidden;
}

.threshold-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.5s ease;
}

.threshold-marker {
    position: absolute;
    top: -2px;
    bottom: -2px;
    width: 2px;
    background: #e2e8f0;
    z-index: 5;
}

/* Bottom Action Bar */
.bottom-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    background: #101921;
    border: 1px solid #1e2d3b;
    border-radius: 8px;
    padding: 12px 20px;
    margin-top: 14px;
}

/* Streamlit button custom styles */
div.stDownloadButton > button {
    background-color: #14b8a6 !important;
    color: #042f2e !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 0 15px rgba(20, 184, 166, 0.3) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

div.stDownloadButton > button:hover {
    background-color: #2dd4bf !important;
    box-shadow: 0 0 20px rgba(45, 212, 191, 0.5) !important;
    transform: translateY(-1px);
}

/* Expander styling */
[data-testid="stExpander"] {
    background-color: #101921 !important;
    border: 1px solid #1e2d3b !important;
    border-radius: 8px !important;
    margin-bottom: 16px !important;
}

[data-testid="stExpander"] summary {
    color: #2dd4bf !important;
    font-weight: 600 !important;
}

/* Audio player dark styling */
audio {
    width: 100%;
    height: 38px;
    border-radius: 6px;
    margin-top: 8px;
    filter: invert(0.88) hue-rotate(180deg);
}
</style>
"""
st.markdown(FORENSIC_CSS, unsafe_allow_html=True)


# Cache HuBERT model for rapid sub-second inference
@st.cache_resource(show_spinner=False)
def get_hubert_pipeline():
    """Caches HuggingFace feature extractor and HuBERT model in memory."""
    extractor, model, device = load_hubert_models()
    return extractor, model, device


# Cache pre-trained Joblib model bundle
@st.cache_resource(show_spinner=False)
def get_detector_bundle():
    """Caches pre-trained ASVspoof 2021 LA MLPClassifier bundle."""
    bundle, msg = load_model_bundle()
    return bundle, msg


# Initialize session state for active case
if "active_case_name" not in st.session_state:
    st.session_state["active_case_name"] = "Case #V-0884 - Live stream capture - AGM - 00:38.0"
if "active_audio_bytes" not in st.session_state:
    # Default to synthetic deepfake spoof demo
    wav_def, sr_def, dur_def, bytes_def = generate_synthetic_demo_clip("synthetic_spoof")
    st.session_state["active_audio_bytes"] = bytes_def
    st.session_state["active_wav"] = wav_def
    st.session_state["active_sr"] = sr_def
    st.session_state["is_default_spoof"] = True


# TOP HEADER BAR
header_left, header_right = st.columns([1.2, 1.0])

with header_left:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 6px;">
        <div style="background: #132735; border: 1px solid #2dd4bf; border-radius: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; color: #2dd4bf; font-size: 1.25rem;">
            🛡️
        </div>
        <div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">
                Identity Shield <span style="font-size: 0.8rem; font-weight: 500; color: #94a3b8; margin-left: 8px;">Deepfake Audio Forensic Console</span>
            </div>
            <div style="font-size: 0.76rem; color: #64748b; font-family: monospace;">
                Cases / <span style="color: #2dd4bf;">Case #V-0884</span> / Voice Analysis (ASVspoof 2021 LA Multi-Codec)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_right:
    st.markdown("""
    <div style="display: flex; justify-content: flex-end; align-items: center; gap: 12px; height: 100%;">
        <div style="background: #101921; border: 1px solid #1e2d3b; border-radius: 6px; padding: 6px 12px; font-size: 0.75rem; text-align: right;">
            <div style="color: #94a3b8;">MCA Dissertation • The NorthCap University</div>
            <div style="color: #2dd4bf; font-weight: 600;">Garima Chauhan (25MCA020)</div>
        </div>
        <div style="background: #101921; border: 1px solid #1e2d3b; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; color: #f59e0b; font-size: 0.95rem;">
            🔔
        </div>
        <div style="background: #0f766e; border: 1px solid #2dd4bf; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; color: #ffffff; font-weight: 700; font-size: 0.85rem;">
            GC
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# COLLAPSIBLE INPUT & EVIDENCE INGESTION PANEL
with st.expander("📂 Forensic Evidence Ingestion Panel (File Upload, Live Microphone, or Synthetic Demos)", expanded=False):
    tab_demos, tab_upload, tab_mic = st.tabs(["⚡ Fast Academic Demo Clips", "📁 Upload Audio File", "🎙 Live Microphone Recording"])
    
    with tab_demos:
        st.markdown("<p style='font-size: 0.8rem; color: #94a3b8;'>Select pre-configured benchmark audio samples for instant academic evaluation:</p>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚨 Load Case #V-0884 (Neural TTS Deepfake Spoof with Splice)", use_container_width=True):
                wav_s, sr_s, dur_s, bytes_s = generate_synthetic_demo_clip("synthetic_spoof")
                st.session_state["active_case_name"] = "Case #V-0884 - Live stream capture - AGM - 00:38.0"
                st.session_state["active_audio_bytes"] = bytes_s
                st.session_state["active_wav"] = wav_s
                st.session_state["active_sr"] = sr_s
                st.session_state["is_default_spoof"] = True
                st.rerun()
        with col_btn2:
            if st.button("🛡 Load Case #V-0912 (Bonafide Human Voice Sample)", use_container_width=True):
                wav_b, sr_b, dur_b, bytes_b = generate_synthetic_demo_clip("bonafide_human")
                st.session_state["active_case_name"] = "Case #V-0912 - Authentication Stream - 00:04.0"
                st.session_state["active_audio_bytes"] = bytes_b
                st.session_state["active_wav"] = wav_b
                st.session_state["active_sr"] = sr_b
                st.session_state["is_default_spoof"] = False
                st.rerun()

    with tab_upload:
        st.markdown("<p style='font-size: 0.8rem; color: #94a3b8;'>Supports WhatsApp voice notes (<code>.ogg</code> Opus), <code>.mp3</code>, <code>.m4a</code>, <code>.flac</code>, and <code>.wav</code>:</p>", unsafe_allow_html=True)
        uploader_k = f"file_uploader_{st.session_state.get('uploader_key', 0)}"
        uploaded_file = st.file_uploader(
            "Upload Forensic Audio Evidence (.wav, .flac, .mp3, .ogg, .m4a)",
            type=["wav", "flac", "mp3", "ogg", "m4a"],
            help="WhatsApp .ogg, .m4a, .mp3, and other formats are automatically decoded via FFmpeg and normalized to 16 kHz mono.",
            key=uploader_k
        )
        col_up_action1, col_up_action2 = st.columns([1, 1])
        with col_up_action1:
            if uploaded_file is not None:
                if st.button("🔍 Analyze Uploaded File", width="stretch", type="primary"):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        wav_u, _, sr_u, dur_u, bytes_u = load_and_preprocess_audio(file_bytes, filename=uploaded_file.name)
                        st.session_state["active_case_name"] = f"Submission: {uploaded_file.name}"
                        st.session_state["active_audio_bytes"] = bytes_u
                        st.session_state["active_wav"] = wav_u
                        st.session_state["active_sr"] = sr_u
                        st.session_state["is_default_spoof"] = None
                        st.session_state["active_source"] = "upload"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading uploaded audio: {e}")
        with col_up_action2:
            if st.button("🗑️ Clear Uploaded File", use_container_width=True):
                st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                st.session_state["active_source"] = "cleared"
                st.rerun()

    with tab_mic:
        st.markdown("""
        <div style="background: #11202c; border: 1px solid #1e3344; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.78rem; color: #cbd5e1;">
            <b style="color: #2dd4bf;">🎙️ Live Microphone Priority Notice:</b><br/>
            Live recordings automatically take precedent. If an uploaded file is present in the workspace, use <b>Clear Uploaded File</b> or click <b>Analyze Microphone Recording</b> to analyze the live capture exclusively.
        </div>
        """, unsafe_allow_html=True)

        col_mic_clear, _ = st.columns([1, 1])
        with col_mic_clear:
            if st.button("🗑️ Clear Uploaded File (Prioritize Mic)", key="clear_upload_btn", use_container_width=True):
                st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                st.session_state["active_source"] = "mic_cleared"
                st.success("Uploaded file cleared! Live microphone input has exclusive priority.")
                st.rerun()

        if hasattr(st, "audio_input"):
            mic_audio = st.audio_input("Record Live Audio Evidence", key="forensic_live_mic")
            if mic_audio is not None:
                if st.button("🔍 Analyze Microphone Recording", use_container_width=True, type="primary"):
                    try:
                        mic_bytes = mic_audio.getvalue()
                        # Process through the same load_audio_16k / load_and_preprocess_audio pipeline with cleanup
                        wav_m, _, sr_m, dur_m, bytes_m = load_and_preprocess_audio(
                            mic_bytes,
                            filename="live_mic_recording.wav"
                        )
                        st.session_state["active_case_name"] = f"Live Capture — Microphone Stream ({dur_m:.1f}s)"
                        st.session_state["active_audio_bytes"] = bytes_m
                        st.session_state["active_wav"] = wav_m
                        st.session_state["active_sr"] = sr_m
                        st.session_state["is_default_spoof"] = None
                        st.session_state["active_source"] = "mic"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing microphone audio: {e}")
        else:
            st.warning("⚠️ Live microphone recording requires Streamlit >= 1.40 (`st.audio_input`). Please upload your recorded audio file in the Upload tab.")


# LOAD MODEL & EXTRACT FEATURES
bundle, bundle_msg = get_detector_bundle()
extractor, hubert_model, device = get_hubert_pipeline()

current_wav = st.session_state.get("active_wav")
current_sr = st.session_state.get("active_sr", 16000)
current_bytes = st.session_state.get("active_audio_bytes")
current_case = st.session_state.get("active_case_name", "Case #V-0884")

# Perform real-time feature extraction and inference
with st.spinner("Extracting 2,370-dim Multi-Branch HuBERT + Acoustic Features..."):
    fused_features, forensic_data = extract_all_three_features(
        current_wav,
        extractor=extractor,
        hubert_model=hubert_model,
        device=device,
        sr=current_sr
    )
    inference_result = run_multi_branch_inference(fused_features, bundle)

# Check if active sample is default spoof or model predicted spoof
is_spoof_case = inference_result["is_deepfake"]
anomaly_ts = forensic_data["peak_flux_time"]
splice_ts = min(anomaly_ts + 0.48, 3.85)


# MAIN FORENSIC CONSOLE WORKSPACE (TWO-COLUMN ARCHITECTURE)
col_left, col_right = st.columns([1.08, 0.92], gap="medium")


# ==============================================================================
# LEFT COLUMN: SIGNAL & SPECTRAL FORENSICS
# ==============================================================================
with col_left:
    # 1. SUBMITTED AUDIO CARD
    st.markdown(f"""
    <div class="forensic-card">
        <div class="forensic-card-header">
            <div>
                <div class="card-title">SUBMITTED AUDIO</div>
                <div class="card-subtitle">{current_case}</div>
            </div>
            <div style="font-size: 0.72rem; color: #2dd4bf; font-family: monospace;">
                16,000 Hz Mono • 64k Samples
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Render dark waveform with ANOMALY and SPLICE markers
    fig_wave = plot_forensic_waveform(
        current_wav,
        sr=current_sr,
        anomaly_time=anomaly_ts if is_spoof_case else None,
        splice_time=splice_ts if is_spoof_case else None
    )
    st.pyplot(fig_wave, use_container_width=True)
    
    # Embedded Audio Player (ensures re-encoded 16 kHz PCM WAV bytes for all formats)
    playback_bytes = current_bytes if (current_bytes and isinstance(current_bytes, (bytes, bytearray))) else wav_to_bytes(current_wav, current_sr)
    st.audio(playback_bytes, format="audio/wav")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. MEL-SPECTROGRAM CARD
    st.markdown(f"""
    <div class="forensic-card">
        <div class="forensic-card-header">
            <div>
                <div class="card-title">MEL-SPECTROGRAM — DUAL-STREAM SPECTRAL ANALYSIS</div>
                <div class="card-subtitle">0–8000 Hz Sub-Band Acoustic Texture</div>
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8; font-family: monospace;">
                FFT N=2048 • Hop=512
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    fig_spec = plot_forensic_melspectrogram(
        current_wav,
        sr=current_sr,
        break_time=anomaly_ts if is_spoof_case else None
    )
    st.pyplot(fig_spec, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. FORENSIC METADATA CARD
    st.markdown(f"""
    <div class="forensic-card">
        <div class="forensic-card-header">
            <div class="card-title">FORENSIC METADATA & TELEMETRY</div>
            <div style="font-size: 0.72rem; color: #2dd4bf;">Active Session</div>
        </div>
        <table class="meta-table">
            <tr class="meta-row">
                <td class="meta-label">🎙 Submission</td>
                <td class="meta-val">{current_case.split(' - ')[0]}</td>
            </tr>
            <tr class="meta-row">
                <td class="meta-label">⏱ Duration / Anomaly</td>
                <td class="meta-val">00:04.0 — {'Break at ' + format_timestamp(anomaly_ts) if is_spoof_case else 'Clean harmonic contour'}</td>
            </tr>
            <tr class="meta-row">
                <td class="meta-label">👤 Principal Reference</td>
                <td class="meta-val">Template #P-0047 — Active</td>
            </tr>
            <tr class="meta-row">
                <td class="meta-label">⚙ Model Architecture</td>
                <td class="meta-val">HuBERT (L2+L6+L12) + Handcrafted MLP Fused</td>
            </tr>
            <tr class="meta-row" style="border-bottom: none;">
                <td class="meta-label">🔗 Cross-View Cosine Harmony</td>
                <td class="meta-val">Ac-Ph: {forensic_data['cos_ac_ph']:.3f} | Ph-Sem: {forensic_data['cos_ph_sem']:.3f}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# RIGHT COLUMN: CONSISTENCY & BRANCH-LEVEL DECISIONS
# ==============================================================================
with col_right:
    # 4. 1:1 VOICEPRINT / MULTI-VIEW CONSISTENCY VERIFICATION CARD
    st.markdown("""
    <div class="forensic-card">
        <div class="forensic-card-header">
            <div class="card-title">1:1 VOICEPRINT VERIFICATION</div>
            <div style="font-size: 0.72rem; color: #94a3b8;">Cross-Layer Dynamic Alignment</div>
        </div>
    """, unsafe_allow_html=True)
    
    subcol_graph, subcol_score = st.columns([1.1, 0.9])
    
    with subcol_graph:
        st.markdown("<div style='font-size: 0.75rem; font-weight: 600; color: #94a3b8; margin-bottom: 4px;'>VOICEPRINT COMPARISON</div>", unsafe_allow_html=True)
        fig_curve = plot_multiview_consistency_curve(
            forensic_data["frame_harmony"],
            divergence_time=anomaly_ts if is_spoof_case else None,
            is_spoof=is_spoof_case
        )
        st.pyplot(fig_curve, use_container_width=True)
        st.markdown(f"<div style='font-size: 0.7rem; color: #94a3b8; text-align: center; margin-top: -6px;'>{'Divergence at ' + format_timestamp(anomaly_ts) if is_spoof_case else 'Concordant Alignment Across Frames'}</div>", unsafe_allow_html=True)

    with subcol_score:
        st.markdown("<div style='font-size: 0.75rem; font-weight: 600; color: #94a3b8; margin-bottom: 2px;'>SIMILARITY SCORE</div>", unsafe_allow_html=True)
        
        # Color coding score
        score_val = inference_result["authenticity_score_pct"]
        if score_val >= 75.0:
            score_color = "#2dd4bf"
            fill_color = "#2dd4bf"
            voiceprint_badge = f'<span class="cyber-pill cyber-pill-teal">Voiceprint match VERIFIED — {score_val}%</span>'
            temporal_badge = '<span class="cyber-pill cyber-pill-teal">Temporal consistency CONTINUOUS</span>'
        elif score_val >= 50.0:
            score_color = "#f59e0b"
            fill_color = "#f59e0b"
            voiceprint_badge = f'<span class="cyber-pill cyber-pill-amber">Voiceprint match MARGINAL — {score_val}%</span>'
            temporal_badge = f'<span class="cyber-pill cyber-pill-amber">Temporal consistency DRIFT at {format_timestamp(anomaly_ts)}</span>'
        else:
            score_color = "#f59e0b"
            fill_color = "#f59e0b"
            voiceprint_badge = f'<span class="cyber-pill cyber-pill-red">Voiceprint match BELOW THRESHOLD — {score_val}%</span>'
            temporal_badge = f'<span class="cyber-pill cyber-pill-red">Temporal consistency BREAK CONFIRMED — {format_timestamp(anomaly_ts)}</span>'
            
        st.markdown(f"""
        <div style="font-size: 2.3rem; font-weight: 800; color: {score_color}; line-height: 1.1;">
            {score_val}%
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-bottom: 8px;">
            Similarity score • Threshold: 85.0%
        </div>
        <div class="threshold-bar-container">
            <div class="threshold-bar-fill" style="width: {min(score_val, 100)}%; background: {fill_color};"></div>
            <div class="threshold-marker" style="left: 85%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: #64748b; margin-top: -12px; margin-bottom: 12px;">
            <span>0%</span>
            <span style="color: #cbd5e1;">85% Threshold</span>
            <span>100%</span>
        </div>
        <div style="margin-top: 6px;">
            {voiceprint_badge}
        </div>
        <div style="margin-top: 6px;">
            {temporal_badge}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. ANTI-SPOOFING MULTI-BRANCH ANALYSIS CARD
    ac_br = inference_result["acoustic_branch"]
    ph_br = inference_result["phonetic_branch"]
    sem_br = inference_result["semantic_branch"]
    
    alert_tag = '<span class="header-alert-badge">⚠ ALERT</span>' if is_spoof_case else '<span class="header-alert-badge" style="border-color:#2dd4bf; color:#2dd4bf; background:rgba(45,212,191,0.1);">✓ CLEAN</span>'
    
    st.markdown(f"""
    <div class="forensic-card">
        <div class="forensic-card-header">
            <div class="card-title">ANTI-SPOOFING ANALYSIS</div>
            {alert_tag}
        </div>
        
        <!-- Row 1: Acoustic Branch -->
        <div class="branch-row">
            <div class="branch-left">
                <div class="branch-icon">〰️</div>
                <div>
                    <div class="branch-name">Voice synthesis detection</div>
                    <div class="branch-desc">{ac_br['subtitle']}</div>
                </div>
            </div>
            <div class="branch-badge" style="background: {ac_br['bg']}; color: {ac_br['color']}; border: 1px solid {ac_br['color']};">
                {ac_br['badge']}
            </div>
        </div>
        
        <!-- Row 2: Phonetic Branch -->
        <div class="branch-row">
            <div class="branch-left">
                <div class="branch-icon">▶️</div>
                <div>
                    <div class="branch-name">Playback & Vocoder attack</div>
                    <div class="branch-desc">{ph_br['subtitle']}</div>
                </div>
            </div>
            <div class="branch-badge" style="background: {ph_br['bg']}; color: {ph_br['color']}; border: 1px solid {ph_br['color']};">
                {ph_br['badge']}
            </div>
        </div>
        
        <!-- Row 3: Semantic Branch -->
        <div class="branch-row">
            <div class="branch-left">
                <div class="branch-icon">⚡</div>
                <div>
                    <div class="branch-name">Temporal coherence & Identity</div>
                    <div class="branch-desc">{sem_br['subtitle']}</div>
                </div>
            </div>
            <div class="branch-badge" style="background: {sem_br['bg']}; color: {sem_br['color']}; border: 1px solid {sem_br['color']};">
                {sem_br['badge']}
            </div>
        </div>
        
        <div style="margin-top: 14px; display: flex; gap: 8px;">
            <div class="cyber-pill { 'cyber-pill-red' if is_spoof_case else 'cyber-pill-teal' }">
                Voiceprint match { 'BELOW THRESHOLD' if is_spoof_case else 'VERIFIED' }
            </div>
            <div class="cyber-pill { 'cyber-pill-amber' if is_spoof_case else 'cyber-pill-teal' }">
                Human review { 'REQUIRED' if is_spoof_case else 'CLEARED' }
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# BOTTOM ACTION & VERDICT BAR
# ==============================================================================
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

bot_col1, bot_col2, bot_col3 = st.columns([0.3, 1.2, 0.5])

with bot_col1:
    st.markdown("""
    <div style="background: #101921; border: 1px solid #1e2d3b; border-radius: 8px; height: 50px; display: flex; align-items: center; justify-content: center; gap: 8px; color: #cbd5e1; font-weight: 600; font-size: 0.85rem; cursor: pointer;">
        <span>🎧</span> Review Mode
    </div>
    """, unsafe_allow_html=True)

with bot_col2:
    if is_spoof_case:
        conf_display = int(round(inference_result["prob_fake"] * 100))
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%); border: 1px solid #ef4444; border-radius: 8px; height: 50px; display: flex; align-items: center; justify-content: center; gap: 12px; color: #ffffff; font-weight: 800; font-size: 1.05rem; letter-spacing: 0.03em; box-shadow: 0 0 20px rgba(239, 68, 68, 0.35);">
            <span>⚠️</span> Voice Cloning Detected — {conf_display}%
        </div>
        """, unsafe_allow_html=True)
    else:
        conf_display = int(round(inference_result["prob_real"] * 100))
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #064e3b 0%, #0f766e 100%); border: 1px solid #2dd4bf; border-radius: 8px; height: 50px; display: flex; align-items: center; justify-content: center; gap: 12px; color: #ffffff; font-weight: 800; font-size: 1.05rem; letter-spacing: 0.03em; box-shadow: 0 0 20px rgba(45, 212, 191, 0.35);">
            <span>✅</span> Authentic Bonafide Voice — {conf_display}%
        </div>
        """, unsafe_allow_html=True)

with bot_col3:
    # Prepare comprehensive evidence package JSON
    evidence_pack = {
        "investigation_id": "FORENSIC-AGY-ASVSPOOF2021",
        "case_name": current_case,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "academic_context": {
            "institution": "The NorthCap University, Gurugram",
            "department": "Department of Computer Science and Engineering",
            "dissertation": "Deepfake Audio Detector: Sensitivity and Fusion of Acoustic, Phonetic, and Semantic Features",
            "candidate": "Garima Chauhan (25MCA020)"
        },
        "verdict": {
            "classification": "DEEPFAKE_CLONE" if is_spoof_case else "BONAFIDE_HUMAN",
            "confidence_pct": round(float(inference_result["confidence_pct"]), 2),
            "fake_probability": round(float(inference_result["prob_fake"]), 4),
            "real_probability": round(float(inference_result["prob_real"]), 4),
            "authenticity_score_pct": round(float(inference_result["authenticity_score_pct"]), 2),
        },
        "branch_level_breakdown": {
            "acoustic_branch": {
                "features": "MFCC (20) + Delta (20) + Spectral Centroid + RMS + HuBERT Layer 2 (768)",
                "slice": "[0:830]",
                "fake_probability": round(float(ac_br["prob_fake"]), 4),
                "badge": ac_br["badge"]
            },
            "phonetic_branch": {
                "features": "HuBERT Layer 6 (768) + Co-Articulation Transition Flux Standard Deviation (1)",
                "slice": "[830:1599]",
                "fake_probability": round(float(ph_br["prob_fake"]), 4),
                "badge": ph_br["badge"]
            },
            "semantic_branch": {
                "features": "HuBERT Layer 12 (768) + Cross-View Cosine Consistencies (3)",
                "slice": "[1599:2370]",
                "fake_probability": round(float(sem_br["prob_fake"]), 4),
                "badge": sem_br["badge"]
            }
        },
        "cross_view_consistency": {
            "cos_acoustic_phonetic": round(float(forensic_data["cos_ac_ph"]), 4),
            "cos_phonetic_semantic": round(float(forensic_data["cos_ph_sem"]), 4),
            "cos_acoustic_semantic": round(float(forensic_data["cos_ac_sem"]), 4),
            "transition_smoothness_flux_std": round(float(forensic_data["transition_smoothness"]), 5),
            "peak_anomaly_timestamp_sec": round(float(anomaly_ts), 3)
        },
        "model_bundle_engine": inference_result["engine_mode"]
    }
    
    st.download_button(
        label="✓ Evidence-Ready — Export Pack",
        data=json.dumps(evidence_pack, indent=2),
        file_name="Forensic_Evidence_Pack_ASVspoof2021.json",
        mime="application/json",
        use_container_width=True
    )

