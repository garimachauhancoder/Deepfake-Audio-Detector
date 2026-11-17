"""
Preprocessing Module for Multi-Branch Speech Deepfake Detection System
Handles audio loading, 16kHz resampling, mono conversion, 4-second normalization,
universal format decoding (WhatsApp .ogg/Opus, .mp3, .m4a, .flac, .wav),
and synthetic demo audio synthesis.
"""

import io
import os
import wave
import shutil
import tempfile
import subprocess
import numpy as np
import soundfile as sf
import librosa


def get_ffmpeg_executable():
    """
    Locates ffmpeg executable from system PATH or bundled imageio_ffmpeg binary.
    """
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None
    return ffmpeg_exe


def wav_to_bytes(wav_array, sr=16000):
    """
    Converts float32 numpy audio array to standard 16-bit PCM WAV bytes.
    Ensures universal playback in Streamlit st.audio(..., format="audio/wav").
    """
    bio = io.BytesIO()
    clamped = np.clip(wav_array, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())
        
    return bio.getvalue()


# Alias for backward compatibility
convert_to_wav_bytes = wav_to_bytes


def load_audio_16k(audio_bytes, filename=None):
    """
    Loads audio from bytes, file-like object, or file path, converting to 16 kHz mono float32.
    Supports WhatsApp .ogg (Opus), .mp3, .m4a, .flac, and .wav without decoding errors:
      1. Preserves original file extension (e.g. .ogg, .m4a, .mp3) in temporary input file.
      2. Tries librosa.load(tmp_path, sr=16000, mono=True) first.
      3. Automatically falls back to ffmpeg (via shutil.which or imageio_ffmpeg.get_ffmpeg_exe())
         to convert any challenging audio codec into 16 kHz mono PCM WAV.
      4. Always cleans up temporary files in finally blocks.
      
    Args:
        audio_bytes: bytes, bytearray, BytesIO, UploadedFile, or file path string
        filename: Optional filename string to extract original extension
        
    Returns:
        wav (np.ndarray): 1D float32 array resampled to 16,000 Hz mono
        sr (int): 16000
    """
    tmp_in_path = None
    tmp_out_path = None

    try:
        # Determine original file extension
        if filename:
            ext = os.path.splitext(filename)[1].lower()
        elif hasattr(audio_bytes, "name"):
            ext = os.path.splitext(audio_bytes.name)[1].lower()
        elif isinstance(audio_bytes, str):
            ext = os.path.splitext(audio_bytes)[1].lower()
        else:
            ext = ".wav"

        if not ext:
            ext = ".wav"

        # Extract raw byte content
        if hasattr(audio_bytes, "getvalue"):
            raw_data = audio_bytes.getvalue()
        elif hasattr(audio_bytes, "read"):
            if hasattr(audio_bytes, "seek"):
                audio_bytes.seek(0)
            raw_data = audio_bytes.read()
        elif isinstance(audio_bytes, (bytes, bytearray)):
            raw_data = bytes(audio_bytes)
        elif isinstance(audio_bytes, str) and os.path.exists(audio_bytes):
            with open(audio_bytes, "rb") as f:
                raw_data = f.read()
        else:
            raise ValueError(f"Unsupported audio source type: {type(audio_bytes)}")

        # Write to temporary file with preserved extension
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
            tmp_in.write(raw_data)
            tmp_in_path = tmp_in.name

        wav = None
        sr = 16000

        # Attempt 1: Direct librosa loading (handles wav, flac, standard mp3, standard ogg)
        try:
            wav, _ = librosa.load(tmp_in_path, sr=16000, mono=True)
            if wav is not None and len(wav) > 0:
                return wav.astype(np.float32), 16000
        except Exception:
            # Fall through to ffmpeg fallback
            pass

        # Attempt 2: Soundfile loading
        try:
            data, orig_sr = sf.read(tmp_in_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if orig_sr != 16000:
                data = librosa.resample(data, orig_sr=orig_sr, target_sr=16000)
            if data is not None and len(data) > 0:
                return data.astype(np.float32), 16000
        except Exception:
            # Fall through to ffmpeg fallback
            pass

        # Attempt 3: Robust ffmpeg conversion (essential for WhatsApp Opus .ogg, .m4a, AAC, etc.)
        ffmpeg_exe = get_ffmpeg_executable()
        if not ffmpeg_exe:
            raise RuntimeError(
                "Failed to decode audio with librosa/soundfile, and ffmpeg executable was not found. "
                "Please ensure imageio-ffmpeg is installed in your virtual environment."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        # ffmpeg command: convert input to 16kHz mono PCM 16-bit WAV
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", tmp_in_path,
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            tmp_out_path
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg audio decoding failed: {err_msg[:300]}")

        data, orig_sr = sf.read(tmp_out_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if orig_sr != 16000:
            data = librosa.resample(data, orig_sr=orig_sr, target_sr=16000)

        return data.astype(np.float32), 16000

    finally:
        # Ensure all temporary files are cleaned up reliably
        if tmp_in_path and os.path.exists(tmp_in_path):
            try:
                os.remove(tmp_in_path)
            except Exception:
                pass
        if tmp_out_path and os.path.exists(tmp_out_path):
            try:
                os.remove(tmp_out_path)
            except Exception:
                pass


def load_and_preprocess_audio(audio_source, filename=None, target_sr=16000, target_duration=4.0):
    """
    Loads audio from a file path, file-like object, or raw bytes via load_audio_16k,
    normalizes/pads/truncates to exactly target_duration (4.0s = 64,000 samples)
    matching the ASVspoof 2021 training pipeline.
    
    Returns:
        wav_4s (np.ndarray): 1D float32 array of shape (64000,)
        wav_raw (np.ndarray): Full original audio resampled to 16kHz mono before pad/truncation
        orig_sr (int): 16000
        orig_duration (float): Original audio duration in seconds
        wav_bytes (bytes): Re-encoded 16 kHz PCM WAV bytes for universal st.audio playback
    """
    if isinstance(audio_source, np.ndarray):
        wav = audio_source.astype(np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        sr = target_sr
    else:
        wav, sr = load_audio_16k(audio_source, filename=filename)

    orig_duration = float(len(wav) / sr)
    wav_raw = wav.copy()

    # Strict 4.0 second (64,000 samples) normalization per ASVspoof 2021 pipeline
    max_len = int(target_sr * target_duration)
    if len(wav) > max_len:
        wav_4s = wav[:max_len]
    else:
        wav_4s = np.pad(wav, (0, max_len - len(wav)), mode="constant")
    
    wav_4s = wav_4s.astype(np.float32)

    # Peak normalization to prevent clipping while preserving dynamics
    max_val = np.max(np.abs(wav_4s))
    if max_val > 1e-4:
        wav_4s = (wav_4s / max_val) * 0.95

    # Always re-encode 16 kHz WAV bytes so the audio player works on every browser and format
    wav_bytes = wav_to_bytes(wav_4s, target_sr)

    return wav_4s, wav_raw, sr, orig_duration, wav_bytes


def generate_synthetic_demo_clip(clip_type="synthetic_spoof", sr=16000, duration=4.0):
    """
    Generates realistic speech-like demo signals:
      - 'synthetic_spoof': Emulates neural TTS / vocoder cloned audio with an unnatural
        phase transition and frequency discontinuity (splice at ~2.84s).
      - 'bonafide_human': Emulates natural voiced speech with human vibrato, formant resonances,
        breathiness, and continuous harmonic coherence.
        
    Returns:
        wav_4s (np.ndarray): 64000-sample audio array
        sr (int): 16000
        duration (float): 4.0
        wav_bytes (bytes): Re-encoded 16 kHz PCM WAV bytes for audio player
    """
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    
    if clip_type == "synthetic_spoof":
        # Multi-formant speech synthesis with neural vocoder artifacts
        f0 = 130.0
        signal = (
            0.45 * np.sin(2 * np.pi * f0 * t) +
            0.25 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.15 * np.sin(2 * np.pi * 3 * f0 * t) +
            0.10 * np.sin(2 * np.pi * 500 * t) * np.exp(-((t % 0.3) / 0.1)) +
            0.08 * np.sin(2 * np.pi * 1500 * t) * np.exp(-((t % 0.25) / 0.08)) +
            0.04 * np.sin(2 * np.pi * 2500 * t) * np.exp(-((t % 0.2) / 0.05))
        )
        
        # Syllabic envelope modulation
        syllable_env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.8 * t)
        signal = signal * syllable_env
        
        # High-frequency vocoder buzz
        vocoder_buzz = 0.03 * np.sin(2 * np.pi * 3800 * t)
        signal += vocoder_buzz
        
        # Temporal break & splice discontinuity at t = 2.84s
        splice_idx = int(2.84 * sr)
        splice_len = int(0.08 * sr)
        if splice_idx + splice_len < total_samples:
            signal[splice_idx : splice_idx + splice_len] = (
                0.8 * np.sin(2 * np.pi * 240 * t[splice_idx : splice_idx + splice_len] + np.pi) +
                0.2 * np.random.normal(0, 0.15, splice_len)
            )
            anomaly_idx = int(3.32 * sr)
            anom_len = int(0.05 * sr)
            if anomaly_idx + anom_len < total_samples:
                signal[anomaly_idx : anomaly_idx + anom_len] += 0.4 * np.sin(2 * np.pi * 880 * t[anomaly_idx : anomaly_idx + anom_len])
                
    else:  # bonafide_human
        f0_contour = 145.0 + 12.0 * np.sin(2 * np.pi * 1.2 * t) + 3.0 * np.sin(2 * np.pi * 5.0 * t)
        phase = 2 * np.pi * np.cumsum(f0_contour) / sr
        
        f1_track = 600.0 + 200.0 * np.sin(2 * np.pi * 0.8 * t)
        f2_track = 1700.0 + 400.0 * np.cos(2 * np.pi * 0.8 * t)
        
        signal = (
            0.40 * np.sin(phase) +
            0.22 * np.sin(2 * phase) +
            0.14 * np.sin(3 * phase) +
            0.08 * np.sin(4 * phase) +
            0.12 * np.sin(2 * np.pi * f1_track * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) +
            0.08 * np.sin(2 * np.pi * f2_track * t) * (0.5 + 0.5 * np.cos(2 * np.pi * 2.5 * t))
        )
        
        pause_env = 1.0 - 0.7 * np.exp(-((t - 2.0) ** 2) / 0.08)
        syllables = np.clip(0.6 + 0.4 * np.sin(2 * np.pi * 2.2 * t) * np.sin(2 * np.pi * 4.4 * t), 0.1, 1.0)
        signal = signal * pause_env * syllables
        
        aspiration = np.random.normal(0, 0.015, total_samples) * (pause_env * syllables)
        signal += aspiration

    peak = np.max(np.abs(signal))
    if peak > 1e-4:
        signal = (signal / peak) * 0.95
        
    signal = signal.astype(np.float32)
    wav_bytes = wav_to_bytes(signal, sr)
    return signal, sr, duration, wav_bytes
