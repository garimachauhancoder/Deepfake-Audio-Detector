# Identity Shield — Multi-Branch Speech Deepfake Detection Platform

**Title**: *"Deepfake Audio Detector: Sensitivity and Fusion of Acoustic, Phonetic, and Semantic Features Under Communication Channel Transformations"*  
**Benchmark**: ASVspoof 2021 Logical Access (LA) Multi-Codec Benchmark (`none`, `opus`, `gsm`, `pstn`, `g722`, `alaw`, `ulaw`)

---

## 1. System Overview & Evidence Taxonomy

`Identity Shield` is an operational cyber-forensic platform for detecting synthetic voice cloning attacks, replay spoofs, and neural text-to-speech (TTS) vocoder artifacts. The system is designed according to a **three-level evidence taxonomy** coupled with a **cross-view consistency framework**:

```
                                  [ Input Audio: 16 kHz Mono, 4.0s (64,000 samples) ]
                                                            │
                     ┌──────────────────────────────────────┼──────────────────────────────────────┐
                     ▼                                      ▼                                      ▼
             [ Acoustic View ]                      [ Phonetic View ]                     [ Semantic View ]
       ┌─────────────────────────────┐        ┌───────────────────────────┐         ┌───────────────────────────┐
       │ Handcrafted: 62 dims        │        │ HuBERT Hidden Layer 6:    │         │ HuBERT Hidden Layer 12:   │
       │ - 20 MFCC means             │        │ - Temporal mean: 768 dims │         │ - Temporal mean: 768 dims │
       │ - 20 MFCC stds              │        │                           │         └─────────────┬─────────────┘
       │ - 20 Delta-MFCC means       │        │ Co-Articulation Cue:      │                       │
       │ - 1 Spectral Centroid mean  │        │ - Frame-to-frame L2 norm  │                       │
       │ - 1 RMS Energy std          │        │   transition std (1 dim)  │                       │
       │                             │        └─────────────┬─────────────┘                       │
       │ HuBERT Hidden Layer 2:      │                      │                                     │
       │ - Temporal mean: 768 dims   │                      │                                     │
       └──────────────┬──────────────┘                      │                                     │
                      │ (830 dims)                          │ (769 dims)                          │ (768 dims)
                      └──────────────────────────────┬──────┴─────────────────────────────────────┘
                                                     ▼
                                     [ Cross-View Cosine Consistency ]
                                      - cos(Acoustic, Phonetic) (1 dim)
                                      - cos(Phonetic, Semantic) (1 dim)
                                      - cos(Acoustic, Semantic) (1 dim)
                                                     │
                                                     ▼
                                [ 2,370-Dimensional Fused Feature Vector ]
                                                     │
                                      [ Fitted StandardScaler ]
                                                     │
                                  [ Multi-Layer Perceptron (MLP) ]
                                                     │
                 ┌───────────────────────────────────┼───────────────────────────────────┐
                 ▼                                   ▼                                   ▼
        [ Acoustic Branch ]                 [ Phonetic Branch ]                 [ Semantic Branch ]
         Slice [0:830]                       Slice [830:1599]                    Slice [1599:2367]
                 └───────────────────────────────────┬───────────────────────────────────┘
                                                     ▼
                                [ Fused Ensemble Decision: Real vs Fake ]
```

---

## 2. Directory Structure

```
Deepfake Audio Detection Platform/
├── app.py                                # Main Streamlit Cyber-Forensic Console
├── requirements.txt                      # Project dependency specification
├── run_app.bat                           # 1-Click Windows Batch Launcher
├── run_app.ps1                           # PowerShell Launcher
├── deepfake_detector_asvspoof2021.pkl     # Pre-trained Joblib Model & Scaler Bundle (Root)
├── models/
│   └── deepfake_detector_asvspoof2021.pkl # Canonical Pre-trained Model Bundle
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py                  # 16kHz resampling, 4.0s pad/trunc, demo generation
│   ├── features.py                       # Cached HuBERT loader & 2,370-dim multi-view feature extractor
│   ├── inference.py                      # Multi-branch probability scoring & decision engine
│   └── visualizations.py                 # Dark forensic Matplotlib figures (Waveform, Mel-Spec, Consistency)
├── demo_samples/
│   ├── case_v0884_deepfake.wav           # Deepfake attack demonstration clip
│   └── case_v0912_bonafide.wav           # Bonafide human voice demonstration clip
├── notebooks/
│   └── Deepfake_Audio_Detection.ipynb    # Jupyter Training & Benchmark Analysis Notebook
└── .streamlit/
    └── config.toml                       # Streamlit server & dark theme configuration
```

---

## 3. Quickstart & Installation

### Option A: Using Pre-Configured Virtual Environment (`.venv`)
The dedicated Python 3.12 virtual environment is pre-installed in `.venv`.

Simply run:
```cmd
run_app.bat
```
or in PowerShell:
```powershell
.\run_app.ps1
```

### Option B: Manual Setup
```bash
# 1. Create a Python 3.12 virtual environment
uv venv .venv --python 3.12
# or: python -m venv .venv

# 2. Activate environment
# On Windows:
.venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt
# or: pip install -r requirements.txt

# 4. Launch Application
streamlit run app.py
```

The application will start at `http://localhost:8501`.

---

## 4. Cyber-Forensic Console UI Features

The interface matches the **Identity Shield** forensic cyber-security console layout:

1. **Top Header Bar**:
   - Identity Shield branding with case breadcrumbs (`Cases / Case #V-0884 / Voice Analysis`).
   - Academic credit: *Garima Chauhan (25MCA020) • The NorthCap University*.

2. **Forensic Evidence Ingestion Panel**:
   - **Fast Academic Demo Clips**: 1-click loading of benchmark Case #V-0884 (Deepfake clone) or Case #V-0912 (Bonafide human speech).
   - **Upload Audio File**: Supports WhatsApp voice notes (`.ogg` Opus), `.m4a`, `.mp3`, `.flac`, and `.wav` with robust automatic FFmpeg decoding via `imageio-ffmpeg`.
   - **Live Microphone Recording**: Test your own voice in real-time (`st.audio_input`) with priority management and instant temporary file cleanup.

3. **Left Column (Signal & Spectral Forensics)**:
   - **Submitted Audio Card**: Displays filename, sample rate, duration, interactive `st.audio` player, and dark teal waveform with highlighted `ANOMALY` (amber) and `SPLICE` (red) vertical markers located from peak Layer-6 frame transition flux.
   - **Mel-Spectrogram Card**: 0–8000 Hz Mel-spectrogram rendered with custom linear segmented colormap (`#0b1116` → `#102a38` → `#1f6f78` → `#64b6ac` → `#e2e8f0`) annotated with dashed `TEMPORAL BREAK` indicator.
   - **Forensic Metadata Card**: Structured table of Submission name, Anomaly timestamps, Principal identity, Model architecture, and Cross-view Cosine similarities.

4. **Right Column (Consistency & Branch-Level Decisions)**:
   - **1:1 Voiceprint & Consistency Verification Card**:
     - *Voiceprint Comparison*: Frame-by-frame harmonic curve compared against Bonafide reference profile with shaded amber divergence area.
     - *Similarity Score*: Big score percentage, horizontal threshold meter (85.0% threshold), and status badges (`BELOW THRESHOLD`, `BREAK CONFIRMED`).
   - **Anti-Spoofing Multi-Branch Analysis Card**:
     - *Acoustic Branch (MFCC + Spectral + HuBERT L2)*: `DETECTED 100%` or `CLEAN`.
     - *Phonetic Branch (HuBERT L6 + Co-Articulation)*: `DETECTED 100%` or `CLEAN`.
     - *Semantic Branch (HuBERT L12 + Cosine)*: `DETECTED 100%` or `CLEAN`.

5. **Bottom Action & Verdict Bar**:
   - `[ 🎧 Review Mode ]` button.
   - **Final Verdict Banner**: High-contrast alert banner (Crimson Red for `⚠️ Voice Cloning Detected — 100%`, Emerald Teal for `✅ Authentic Bonafide Voice`).
   - **`✓ Evidence-Ready — Export Pack`**: Streamlit download button generating a structured JSON evidence file with complete mathematical parameters, timestamps, branch scores, and academic metadata.

