import streamlit as st
import random
import time
import tempfile
import os
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import cv2
import warnings
import json
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)

warnings.filterwarnings("ignore")

HF_REPO_ID        = "AI-BUSTER/AI-BUSTER_Models"
HF_AUDIO_FILE     = "audio_model.pth"
HF_VIDEO_FILE     = "video_model.pth"
HF_IMAGE_FILE     = "image_model.pth"
HF_TEXT_SCALER_FILE = "scaler.pkl"
HF_TEXT_STATE_FILE  = "text_model.pth"
HF_FEEDBACK_FILE = "feedback.json"
HF_ANALYTICS_REPO = "YusufCaganCeylan/AI-BUSTER_Analytics"
HF_ANALYTICS_FILE = "analytics.json"
HF_TOKEN          = os.environ.get("HF_TOKEN")
HF_TOKEN_W        = os.environ.get("HF_TOKEN_W")

MIN_WORDS_REQUIRED = 150

# ── Logo yükleme (base64 ile gömme) ──────────────────────────────────────────
LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo.jpg")

def get_logo_base64() -> str:
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return ""

LOGO_B64 = get_logo_base64()

def logo_img_tag(width: int = 42) -> str:
    if LOGO_B64:
        return (
            f'<img src="data:image/jpeg;base64,{LOGO_B64}" '
            f'style="width:{width}px;height:{width}px;border-radius:10px;'
            f'object-fit:cover;flex-shrink:0;" />'
        )
    return '<span style="font-size:1.8rem;line-height:1;">🛡️</span>'

# ── Uzantıdan mod tespiti ──────────────────────────────────────────────────
AUDIO_EXTS  = {"mp3", "wav", "flac", "ogg", "m4a"}
VIDEO_EXTS  = {"mp4", "mov", "avi", "mkv", "webm"}
IMAGE_EXTS  = {"jpg", "jpeg", "png", "webp", "bmp"}
TEXT_EXTS   = {"txt", "pdf", "docx", "md"}
ALL_EXTS    = list(AUDIO_EXTS | VIDEO_EXTS | IMAGE_EXTS | TEXT_EXTS)

def detect_mode_from_ext(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in AUDIO_EXTS:  return "🎵 Ses"
    if ext in VIDEO_EXTS:  return "🎬 Video"
    if ext in IMAGE_EXTS:  return "🖼️ Görsel"
    if ext in TEXT_EXTS:   return "📝 Metin"
    return None

st.set_page_config(
    page_title="AI-BUSTER",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import streamlit.components.v1 as components
components.html("""
<script>
(function() {
  function init() {
    const doc = window.parent.document;
    if (doc.getElementById('aibuster-toggle')) return;

    const btn = doc.createElement('button');
    btn.id = 'aibuster-toggle';
    btn.innerHTML = '&#9776;';
    btn.style.cssText = `
      position: fixed;
      left: 0;
      top: 50%;
      transform: translateY(-50%);
      z-index: 999999;
      background: linear-gradient(135deg, #7C3AED, #06B6D4);
      border: none;
      border-radius: 0 10px 10px 0;
      width: 28px;
      height: 64px;
      color: white;
      font-size: 16px;
      cursor: pointer;
      box-shadow: 4px 0 20px rgba(124,58,237,0.4);
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
    `;

    btn.addEventListener('mouseenter', () => btn.style.width = '34px');
    btn.addEventListener('mouseleave', () => btn.style.width = '28px');

    btn.addEventListener('click', () => {
      const sidebar = doc.querySelector('[data-testid="stSidebar"]');
      const collapseBtn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                          doc.querySelector('[data-testid="collapsedControl"] button') ||
                          doc.querySelector('button[aria-label="Collapse sidebar"]') ||
                          doc.querySelector('button[aria-label="Open sidebar"]') ||
                          doc.querySelector('button[aria-label="Close sidebar"]');

      if (collapseBtn) {
        collapseBtn.click();
      } else if (sidebar) {
        const expanded = sidebar.getAttribute('aria-expanded');
        sidebar.setAttribute('aria-expanded', expanded === 'true' ? 'false' : 'true');
      }
    });

    doc.body.appendChild(btn);
  }

  if (document.readyState === 'complete') {
    setTimeout(init, 500);
  } else {
    window.addEventListener('load', () => setTimeout(init, 500));
  }
  setTimeout(init, 1000);
  setTimeout(init, 2000);
})();
</script>
""", height=0)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Manrope:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
:root {
    --bg:        #131317;
    --surface-0: #0e0e12;
    --surface-1: #1b1b1f;
    --surface-2: #1f1f23;
    --surface-3: #2a292e;
    --surface-4: #353439;
    --primary:   #7C3AED;
    --cyan:      #06B6D4;
    --cyan-dim:  #4cd7f6;
    --on-surface:#e4e1e7;
    --muted:     #ccc3d8;
    --outline:   #4a4455;
    --error:     #ffb4ab;
}
html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--on-surface) !important;
}
#MainMenu, header, footer { display: none !important; }
.stDeployButton { display: none !important; }
section[data-testid="stSidebar"] > div { background-color: var(--surface-0) !important; }
.stApp { background-color: var(--bg) !important; }
[data-testid="stSidebar"] {
    background: var(--surface-0) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: var(--muted) !important;
    font-family: 'Manrope', sans-serif !important;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em !important;
}
[data-testid="metric-container"] {
    background: var(--surface-1) !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
    border-left: 3px solid var(--primary) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.8rem !important;
    color: var(--on-surface) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stFileUploader"] {
    background: var(--surface-1) !important;
    border: 2px dashed rgba(124,58,237,0.3) !important;
    border-radius: 18px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div small,
[data-testid="stFileUploaderDropzoneInstructions"] div span small,
section[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] small { display: none !important; }

[data-testid="stFileUploaderDropzoneInstructions"] span:not(:first-child) { display: none !important; }

[data-testid="stFileUploaderDropzoneInstructions"] span:first-child {
    visibility: visible !important;
    font-size: 0.72rem !important;
    color: #ccc3d8 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.25) !important;
}
.stButton > button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 30px rgba(6,182,212,0.35) !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7C3AED, #06B6D4) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: var(--surface-4) !important;
    border-radius: 4px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface-1) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
    color: white !important;
}
.stTextArea textarea {
    background: var(--surface-1) !important;
    color: var(--on-surface) !important;
    border: 1px solid var(--outline) !important;
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
}
.stSpinner { color: var(--cyan) !important; }
[data-testid="stDataFrame"] {
    background: var(--surface-1) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
[data-baseweb="select"] { background: var(--surface-1) !important; }
[data-baseweb="select"] > div {
    background: var(--surface-1) !important;
    border-color: var(--outline) !important;
    color: var(--on-surface) !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface-0); }
::-webkit-scrollbar-thumb { background: var(--outline); border-radius: 3px; }
/* Sidebar toggle */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="collapsedControl"] svg { color: white !important; fill: white !important; width: 14px !important; height: 14px !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] button { background: transparent !important; border: none !important; color: white !important; cursor: pointer !important; width: 100% !important; height: 100% !important; padding: 0 !important; }
[data-testid="stSidebarCollapseButton"] button svg { color: white !important; fill: white !important; width: 14px !important; height: 14px !important; }
button[aria-label="Close sidebar"] { display: none !important; }
button[aria-label="Collapse sidebar"] { display: none !important; }
section[data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important; }
section[data-testid="stSidebar"][aria-expanded="false"] { min-width: 0px !important; max-width: 0px !important; overflow: hidden !important; }

/* ── Görsel önizleme: layout shift önleme ── */
.aibuster-img-preview {
    width: 100%;
    max-width: 320px;
    margin-top: 0.5rem;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.07);
    display: block;
    height: auto;
    /* Sabit alan ayır — görsel yüklenince layout kaymasın */
    min-height: 180px;
    background: #1b1b1f;
    object-fit: contain;
}
.aibuster-img-wrapper {
    width: 100%;
    max-width: 320px;
    margin-top: 0.5rem;
    /* Taşmayı engelle — re-render'da genişlik değişmesin */
    overflow: hidden;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.07);
    background: #1b1b1f;
    min-height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALİTİK
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_analytics_from_hf() -> dict:
    varsayilan = {"history": [], "total_analyzed": 0, "deepfake_hits": 0}
    try:
        path = hf_hub_download(
            repo_id=HF_ANALYTICS_REPO,
            filename=HF_ANALYTICS_FILE,
            repo_type="dataset",
            token=HF_TOKEN_W,
            force_download=True,
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "total_analyzed" not in data:
            data["total_analyzed"] = len(data.get("history", []))
        if "deepfake_hits" not in data:
            data["deepfake_hits"] = sum(
                1 for h in data.get("history", []) if h.get("ai", 0) >= 70
            )
        return data
    except Exception:
        return varsayilan


if "analytics_loaded" not in st.session_state:
    _data = _fetch_analytics_from_hf()
    st.session_state.history        = _data["history"]
    st.session_state.total_analyzed = _data["total_analyzed"]
    st.session_state.deepfake_hits  = _data["deepfake_hits"]
    st.session_state.analytics_loaded = True


def save_analytics(new_entry: dict, is_deepfake: bool):
    current_history = list(st.session_state.history)
    updated = {
        "history":        current_history[:500],
        "total_analyzed": st.session_state.total_analyzed,
        "deepfake_hits":  st.session_state.deepfake_hits,
    }
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=json.dumps(updated, ensure_ascii=False, indent=2).encode("utf-8"),
            path_in_repo=HF_ANALYTICS_FILE,
            repo_id=HF_ANALYTICS_REPO,
            repo_type="dataset",
            token=HF_TOKEN_W,
            commit_message=f"Analiz eklendi: {new_entry.get('file', '?')}",
        )
    except Exception as e:
        st.warning(f"⚠️ Analitik kaydedilemedi: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SES MODEL MİMARİSİ
# ══════════════════════════════════════════════════════════════════════════════

class AudioConfig:
    SAMPLE_RATE    = 16000
    CHUNK_DURATION = 4
    CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_DURATION
    N_LFCC         = 60
    N_FFT          = 512
    HOP_LENGTH     = 160
    WIN_LENGTH     = 400
    N_FILTERS      = 128
    HIDDEN_DIM     = 256
    DROPOUT        = 0.5
    NUM_CLASSES    = 2
    WAV2VEC_MODEL  = "facebook/wav2vec2-xls-r-300m"


class VoiceActivityDetector:
    def __init__(self, frame_duration_ms=30, energy_threshold=0.02, min_speech_duration_ms=100):
        self.frame_duration_ms      = frame_duration_ms
        self.energy_threshold       = energy_threshold
        self.min_speech_duration_ms = min_speech_duration_ms

    def detect_speech(self, audio, sr):
        frame_length = int(sr * self.frame_duration_ms / 1000)
        hop_length   = frame_length // 2
        energy = np.array([
            np.sum(audio[i:i + frame_length] ** 2)
            for i in range(0, len(audio) - frame_length, hop_length)
        ])
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        speech_frames = energy > self.energy_threshold
        min_frames    = int(self.min_speech_duration_ms / self.frame_duration_ms * 2)
        speech_audio, consecutive, speech_start = [], 0, None
        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                if speech_start is None:
                    speech_start = i
                consecutive += 1
            else:
                if consecutive >= min_frames and speech_start is not None:
                    speech_audio.append(audio[speech_start * hop_length: min(i * hop_length + frame_length, len(audio))])
                speech_start, consecutive = None, 0
        if consecutive >= min_frames and speech_start is not None:
            speech_audio.append(audio[speech_start * hop_length:])
        return np.concatenate(speech_audio) if speech_audio else audio


def chunk_audio(audio, chunk_samples, overlap=0.5):
    hop_samples = int(chunk_samples * (1 - overlap))
    chunks = []
    for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
        chunks.append(audio[start:start + chunk_samples])
    if len(audio) >= chunk_samples // 2:
        remaining = audio[-(len(audio) % chunk_samples):] if len(audio) % chunk_samples != 0 else None
        if remaining is not None and len(remaining) >= chunk_samples // 2:
            chunks.append(np.pad(remaining, (0, chunk_samples - len(remaining)), mode='constant'))
    if not chunks:
        chunks.append(np.pad(audio, (0, chunk_samples - len(audio)), mode='constant'))
    return chunks


class LFCCExtractor:
    def __init__(self, sr=16000, n_lfcc=60, n_fft=512, hop_length=160, win_length=400, n_filters=128):
        self.sr, self.n_lfcc, self.n_fft = sr, n_lfcc, n_fft
        self.hop_length, self.win_length, self.n_filters = hop_length, win_length, n_filters
        self.filter_bank = self._create_linear_filterbank()

    def _create_linear_filterbank(self):
        cf  = np.linspace(0, self.sr // 2, self.n_filters + 2)
        bf  = np.floor((self.n_fft + 1) * cf / self.sr).astype(int)
        fb  = np.zeros((self.n_filters, self.n_fft // 2 + 1))
        for i in range(self.n_filters):
            for j in range(bf[i], bf[i + 1]):
                fb[i, j] = (j - bf[i]) / (bf[i + 1] - bf[i])
            for j in range(bf[i + 1], bf[i + 2]):
                fb[i, j] = (bf[i + 2] - j) / (bf[i + 2] - bf[i + 1])
        return fb

    def extract(self, audio):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        stft     = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                                 win_length=self.win_length, window="hamming")
        lin_spec = np.dot(self.filter_bank, np.abs(stft) ** 2)
        lin_spec = np.where(lin_spec == 0, np.finfo(float).eps, lin_spec)
        lfcc     = librosa.feature.mfcc(S=np.log(lin_spec), n_mfcc=self.n_lfcc, dct_type=2)
        return np.concatenate([lfcc, librosa.feature.delta(lfcc), librosa.feature.delta(lfcc, order=2)], axis=0)


class Wav2VecStream(nn.Module):
    def __init__(self, model_name, hidden_dim=256, dropout=0.5, freeze_feature_extractor=True):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        if freeze_feature_extractor:
            for param in self.wav2vec.feature_extractor.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.wav2vec.encoder.layers):
                if i < len(self.wav2vec.encoder.layers) // 2:
                    for param in layer.parameters():
                        param.requires_grad = False
        wav2vec_dim = self.wav2vec.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(wav2vec_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.output_dim = hidden_dim

    def forward(self, audio):
        out    = self.wav2vec(audio).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


class MFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.mfm1     = MFM(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout)
        self.mfm2     = MFM(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.mfm1(x);  out = self.bn1(out);  out = self.dropout1(out)
        out = self.mfm2(out); out = self.bn2(out);  out = self.dropout2(out)
        if out.shape != identity.shape:
            identity = F.adaptive_avg_pool2d(identity, out.shape[2:])
        return out + identity


class LFCCStream(nn.Module):
    def __init__(self, n_lfcc=60, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.conv1 = MFM(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1   = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(dropout * 0.4)
        self.block1 = LCNNBlock(32,  48,  dropout=dropout * 0.3)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.block2 = LCNNBlock(48,  96,  dropout=dropout * 0.3)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.block3 = LCNNBlock(96,  128, dropout=dropout * 0.3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.output_dim = hidden_dim

    def forward(self, lfcc):
        x = lfcc.unsqueeze(1)
        x = self.conv1(x);  x = self.pool1(x); x = self.bn1(x); x = self.drop1(x)
        x = self.block1(x); x = self.pool2(x)
        x = self.block2(x); x = self.pool3(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)


class AttentionFusion(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, s1, s2):
        w = self.attention(torch.cat([s1, s2], dim=-1))
        return w[:, 0:1] * s1 + w[:, 1:2] * s2, w


class TwoStreamDeepfakeDetector(nn.Module):
    def __init__(self, config, wav2vec_model="facebook/wav2vec2-xls-r-300m"):
        super().__init__()
        self.wav2vec_stream  = Wav2VecStream(wav2vec_model, config.HIDDEN_DIM, config.DROPOUT)
        self.lfcc_stream     = LFCCStream(config.N_LFCC,   config.HIDDEN_DIM, config.DROPOUT)
        self.feature_dropout = nn.Dropout(config.DROPOUT * 0.3)
        self.fusion          = AttentionFusion(config.HIDDEN_DIM, 64, config.DROPOUT * 0.5)
        self.classifier      = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM,      config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.LayerNorm(config.HIDDEN_DIM // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT * 0.5),
            nn.Linear(config.HIDDEN_DIM // 4, config.NUM_CLASSES),
        )

    def forward(self, audio, lfcc):
        w = self.wav2vec_stream(audio)
        l = self.lfcc_stream(lfcc)
        w = self.feature_dropout(w)
        l = self.feature_dropout(l)
        fused, attn = self.fusion(w, l)
        logits = self.classifier(fused)
        probs  = F.softmax(logits, dim=-1)
        return logits, probs, attn


# ══════════════════════════════════════════════════════════════════════════════
#  GÖRSEL MODEL MİMARİSİ
# ══════════════════════════════════════════════════════════════════════════════
class SigLIP2Detector(nn.Module):
    def __init__(self, model_name="google/siglip2-so400m-patch14-384",
                 freeze_backbone=True, dropout_head=0.45, dropout_mid=0.25):
        super().__init__()
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(model_name)
        self.vision_model = backbone.vision_model
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        hidden_dim = self.vision_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout_head),
            nn.Linear(512, 256),            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout_mid),
            nn.Linear(256, 64),             nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout_mid * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, pixel_values):
        out  = self.vision_model(pixel_values=pixel_values,
                                  output_hidden_states=False,
                                  interpolate_pos_encoding=True)
        h    = out.last_hidden_state
        mean = h.mean(dim=1)
        mx   = h.max(dim=1).values
        return self.classifier(torch.cat([mean, mx], dim=1)).squeeze(1)

# ══════════════════════════════════════════════════════════════════════════════
#  METİN MODEL MİMARİSİ
# ══════════════════════════════════════════════════════════════════════════════

import pickle
import re


class TextConfig:
    MODEL_NAME         = "dbmdz/bert-base-turkish-cased"
    MAX_LENGTH         = 256
    NUM_LABELS         = 2
    NUM_STYLE_FEATURES = 12
    AI_THRESHOLD       = 0.85


class HybridBert(nn.Module):
    def __init__(self, model_name, num_style_features, num_labels=2):
        super().__init__()
        from transformers import BertModel
        self.bert   = BertModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.style_mlp = nn.Sequential(
            nn.Linear(num_style_features, 4),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_labels),
        )

    def forward(self, input_ids, attention_mask, style_features, token_type_ids=None):
        out   = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        cls   = out.last_hidden_state[:, 0, :]
        sf    = style_features.to(cls.device).float()
        style_out = self.style_mlp(sf)
        combined  = torch.cat([cls, style_out], dim=1)
        return self.classifier(combined)


def _extract_style_features(text: str) -> np.ndarray:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    words     = text.split()
    num_words = max(len(words), 1)
    num_sents = max(len(sentences), 1)
    sent_lens = [len(s.split()) for s in sentences] or [0]
    avg_sent_len  = float(np.mean(sent_lens))
    std_sent_len  = float(np.std(sent_lens))
    ttr           = len(set(w.lower() for w in words)) / num_words
    comma_rate    = text.count(',') / num_words
    punct_rate    = sum(text.count(p) for p in '.,;:!?') / num_words
    paragraphs    = [p.strip() for p in text.split('\n') if len(p.strip()) > 10]
    num_paras     = len(paragraphs) or 1
    avg_para_len  = float(np.mean([len(p.split()) for p in paragraphs])) if paragraphs else 0.0
    avg_word_len  = float(np.mean([len(w) for w in words]))
    bigrams       = list(zip(words[:-1], words[1:]))
    bigram_repeat = 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    ai_connectors = ['ayrıca', 'bunun yanı sıra', 'öte yandan', 'sonuç olarak',
                     'bu bağlamda', 'bu doğrultuda', 'nitekim', 'dolayısıyla']
    connector_rate = sum(text.lower().count(c) for c in ai_connectors) / num_sents
    feats = [avg_sent_len, std_sent_len, ttr, comma_rate, punct_rate, num_paras,
             avg_para_len, avg_word_len, bigram_repeat, connector_rate, num_words, num_sents]
    return np.array(feats).reshape(1, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  VİDEO MODEL MİMARİSİ
# ══════════════════════════════════════════════════════════════════════════════

EPSILON_V = 1e-8


@dataclass
class TrainingConfig:
    train_dir: str = ""
    val_dir:   str = ""
    test_dir:  str = ""
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 3e-5
    backbone_lr_mult: float = 0.1
    seq_length: int = 12
    img_size: int = 224
    patience: int = 15
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    label_smoothing: float = 0.05
    warmup_epochs: int = 3
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_dropout: float = 0.2
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_hard_mining: bool = True
    hard_mining_ratio: float = 0.7
    use_ema: bool = True
    ema_decay: float = 0.999
    use_gradient_checkpointing: bool = True
    use_optical_flow: bool = False
    device: torch.device = field(init=False)
    mixed_precision: bool = field(init=False)
    num_workers: int = field(init=False)
    gpu_name: str = field(init=False)
    gpu_memory_gb: float = field(init=False)

    def __post_init__(self):
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = torch.cuda.is_available()
        self.gpu_name        = ""
        self.gpu_memory_gb   = 0.0
        self.num_workers     = 2 if platform.system() == "Linux" else 0
        if torch.cuda.is_available():
            self.gpu_name      = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    def auto_scale_for_dataset(self, total_videos: int):
        pass


class VideoNumpyTransform:
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img):
        if img.shape[:2] != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        normed = (img - self.mean) / (self.std + EPSILON_V)
        return torch.from_numpy(normed.transpose(2, 0, 1)).float()


def get_frequency_features(img_rgb):
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gf   = gray.astype(np.float32) / 255.0
        b1 = cv2.GaussianBlur(gf, (3,  3),  0.5)
        b2 = cv2.GaussianBlur(gf, (7,  7),  1.5)
        b3 = cv2.GaussianBlur(gf, (11, 11), 3.0)
        dog = np.abs(b1 - b2) + np.abs(b2 - b3) + np.abs(gf - b1)
        ch0 = dog / (dog.max() + EPSILON_V)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=5))
        ch1 = (lap / (lap.max() + EPSILON_V)).astype(np.float32)
        r, g, b = [img_rgb[:, :, i].astype(np.float32) for i in range(3)]
        gr  = cv2.Sobel(r, cv2.CV_64F, 1, 1, ksize=3)
        gg  = cv2.Sobel(g, cv2.CV_64F, 1, 1, ksize=3)
        gb_ = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3)
        incon = (np.abs(gr - gg) + np.abs(gg - gb_)) / 2.0
        ch2   = (incon / (incon.max() + EPSILON_V)).astype(np.float32)
        return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)
    except Exception:
        return np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.float32)


class FrequencyStreamCNN(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32,  3, 2, 1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32,   64,  3, 2, 1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64,  128,  3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256,  3, 2, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512,  3, 2, 1), nn.BatchNorm2d(512), nn.GELU(),
        )
        self.pool        = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512

    def forward(self, x):
        return self.pool(self.net(x)).flatten(1)


class DualStreamExtractor(nn.Module):
    def __init__(self, use_optical_flow=False):
        super().__init__()
        from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
        eff               = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        self.rgb_features = eff.features
        self.rgb_pool     = eff.avgpool
        freq_in           = 6 if use_optical_flow else 3
        self.freq_stream  = FrequencyStreamCNN(in_ch=freq_in)
        total             = 2048 + self.freq_stream.feature_dim
        self.feature_dim  = 1024
        self.rgb_gate     = nn.Sequential(nn.Linear(total, 1), nn.Sigmoid())
        self.freq_gate    = nn.Sequential(nn.Linear(total, 1), nn.Sigmoid())
        self.fusion       = nn.Sequential(
            nn.Linear(total, self.feature_dim),
            nn.LayerNorm(self.feature_dim), nn.GELU(), nn.Dropout(0.1),
        )
        self.use_flow = use_optical_flow

    def forward(self, x):
        B, S    = x.shape[:2]
        rgb_in  = x[:, :, :3].reshape(B * S, 3, x.size(3), x.size(4))
        freq_in = x[:, :, 3:6]
        if self.use_flow and x.size(2) > 6:
            freq_in = torch.cat([freq_in, x[:, :, 6:9]], dim=2)
        freq_in = freq_in.reshape(B * S, -1, x.size(3), x.size(4))
        rf      = torch.flatten(self.rgb_pool(self.rgb_features(rgb_in)), 1)
        ff      = self.freq_stream(freq_in)
        cat     = torch.cat([rf, ff], dim=1)
        gated   = torch.cat([rf * self.rgb_gate(cat), ff * self.freq_gate(cat)], dim=1)
        return self.fusion(gated).view(B, S, -1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len=32, d_model=1024):
        super().__init__()
        self.pe   = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d=1024, heads=8, layers=6, dropout=0.2, max_seq=32):
        super().__init__()
        self.pos        = LearnablePositionalEncoding(max_seq, d)
        self.diff_proj  = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.diff_alpha = nn.Parameter(torch.tensor(0.1))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf   = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        diff        = torch.zeros_like(x)
        diff[:, 1:] = x[:, 1:] - x[:, :-1]
        x           = x + self.diff_alpha * self.diff_proj(diff)
        return self.norm(self.tf(self.pos(x)))


class ResBlock(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU())
        self.skip = nn.Linear(in_d, out_d) if in_d != out_d else nn.Identity()
        self.norm = nn.LayerNorm(out_d)

    def forward(self, x):
        return self.norm(self.main(x) + self.skip(x))


class DeepfakeDetectorV3(nn.Module):
    def __init__(self, use_optical_flow, transformer_heads, transformer_layers, transformer_dropout):
        super().__init__()
        self.spatial  = DualStreamExtractor(use_optical_flow=use_optical_flow)
        d             = self.spatial.feature_dim
        self.temporal = TemporalTransformerEncoder(d=d, heads=transformer_heads, layers=transformer_layers, dropout=transformer_dropout)
        self.attn_pool  = nn.MultiheadAttention(d, num_heads=8, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            ResBlock(d,   512), nn.Dropout(0.3),
            ResBlock(512, 256), nn.Dropout(0.2),
            ResBlock(256,  64), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feats  = self.spatial(x)
        t      = self.temporal(feats)
        q      = t.mean(dim=1, keepdim=True)
        att, _ = self.attn_pool(q, t, t)
        return self.classifier(att.squeeze(1))


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL YÜKLEME
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_audio_model():
    try:
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_AUDIO_FILE, token=HF_TOKEN)
        config = AudioConfig()
        model  = TwoStreamDeepfakeDetector(config, config.WAV2VEC_MODEL).to(device)
        ckpt   = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict):
            key   = next((k for k in ("model_state_dict", "state_dict") if k in ckpt), None)
            state = ckpt[key] if key else ckpt
        else:
            state = ckpt
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
        model.eval()
        return model, config, device, None
    except Exception as e:
        import traceback
        return None, None, None, traceback.format_exc()


@st.cache_resource(show_spinner=False)
def get_video_model():
    try:
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_VIDEO_FILE, token=HF_TOKEN)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", None)
        def g(attr, default):
            return getattr(cfg, attr, default) if cfg is not None else default
        video_cfg = {
            "seq_length": g("seq_length", 12), "img_size": g("img_size", 224),
            "use_optical_flow": g("use_optical_flow", False),
            "transformer_heads": g("transformer_heads", 8),
            "transformer_layers": g("transformer_layers", 6),
            "transformer_dropout": g("transformer_dropout", 0.2),
            "val_acc": ckpt.get("val_acc", None), "val_f1": ckpt.get("f1", None),
            "val_auc": ckpt.get("auc", None),
        }
        model = DeepfakeDetectorV3(
            use_optical_flow=video_cfg["use_optical_flow"],
            transformer_heads=video_cfg["transformer_heads"],
            transformer_layers=video_cfg["transformer_layers"],
            transformer_dropout=video_cfg["transformer_dropout"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device); model.eval()
        return model, video_cfg, device, None
    except Exception as e:
        import traceback
        return None, None, None, traceback.format_exc()


@st.cache_resource(show_spinner=False)
def get_image_model():
    try:
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_IMAGE_FILE, token=HF_TOKEN)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg          = ckpt.get("config", {}) or {}
        dropout_head = ckpt.get("dropout_head", cfg.get("dropout_head", 0.45))
        dropout_mid  = ckpt.get("dropout_mid",  cfg.get("dropout_mid",  0.25))
        model_name   = cfg.get("model_name", "google/siglip2-so400m-patch14-384")
        image_size   = cfg.get("image_size", 384)
        model = SigLIP2Detector(model_name=model_name, freeze_backbone=True,
                                dropout_head=dropout_head, dropout_mid=dropout_mid)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device); model.eval()
        image_cfg = {
            "model_name":    cfg.get("model_name", "google/siglip2-so400m-patch14-384"),
            "image_size":    cfg.get("image_size", 224),
            "val_acc":       ckpt.get("val_acc",   None),
            "val_f1":        ckpt.get("best_f1",   None),
            "val_auc":       ckpt.get("val_auc",   None),
            "val_precision": ckpt.get("test_precision", None),
            "val_recall":    ckpt.get("test_recall",    None),
            "dropout_head":  cfg.get("dropout_head", 0.50),
            "dropout_mid":   cfg.get("dropout_mid",  0.30),
        }
        return model, image_cfg, device, None
    except Exception as e:
        import traceback
        return None, None, None, traceback.format_exc()


@st.cache_resource(show_spinner=False)
def get_text_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = TextConfig()
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        scaler_path     = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_TEXT_SCALER_FILE, token=HF_TOKEN)
        state_dict_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_TEXT_STATE_FILE, token=HF_TOKEN)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        model = HybridBert(
            model_name=config.MODEL_NAME,
            num_style_features=config.NUM_STYLE_FEATURES,
            num_labels=config.NUM_LABELS,
        ).to(device)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        model.eval()
        return model, tokenizer, scaler, config, device, None
    except Exception as e:
        import traceback
        return None, None, None, None, None, traceback.format_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_audio(audio_path, model, config, device):
    audio, _     = librosa.load(audio_path, sr=config.SAMPLE_RATE)
    speech_audio = VoiceActivityDetector().detect_speech(audio, config.SAMPLE_RATE)
    chunks       = chunk_audio(speech_audio, config.CHUNK_SAMPLES)
    lfcc_ext     = LFCCExtractor(config.SAMPLE_RATE, config.N_LFCC, config.N_FFT,
                                  config.HOP_LENGTH, config.WIN_LENGTH, config.N_FILTERS)
    lfcc_feats   = [lfcc_ext.extract(c) for c in chunks]
    all_preds, all_probs, all_attn = [], [], []
    model.eval()
    with torch.no_grad():
        for chunk, lfcc in zip(chunks, lfcc_feats):
            logits, probs, attn = model(
                torch.FloatTensor(chunk).unsqueeze(0).to(device),
                torch.FloatTensor(lfcc).unsqueeze(0).to(device),
            )
            all_preds.append(torch.argmax(probs, dim=1).item())
            all_probs.append([probs[0, 0].item(), probs[0, 1].item()])
            all_attn.append(attn[0].cpu().numpy())
    arr        = np.array(all_probs)
    avg_real   = float(arr[:, 0].mean())
    avg_fake   = float(arr[:, 1].mean())
    attn_arr   = np.array(all_attn)
    confidence = max(avg_real, avg_fake) * 100
    conf_level = ("ÇOK YÜKSEK" if confidence > 90 else "YÜKSEK" if confidence > 75 else "ORTA" if confidence > 60 else "DÜŞÜK")
    return {
        "fake_pct": round(avg_fake * 100, 2), "real_pct": round(avg_real * 100, 2),
        "label": "SAHTE" if avg_fake > avg_real else "GERÇEK",
        "confidence": round(confidence, 2), "confidence_level": conf_level,
        "chunk_count": len(chunks), "fake_chunks": sum(all_preds),
        "real_chunks": len(all_preds) - sum(all_preds),
        "wav2vec_attn": round(float(attn_arr[:, 0].mean()), 3),
        "lfcc_attn": round(float(attn_arr[:, 1].mean()), 3),
        "duration_sec": round(len(audio) / config.SAMPLE_RATE, 2),
        "vad_duration_sec": round(len(speech_audio) / config.SAMPLE_RATE, 2),
        "scores": {
            "Sahte Olasılığı":  round(avg_fake * 100, 1),
            "Gerçek Olasılığı": round(avg_real * 100, 1),
        },
    }


def _video_to_tensor(video_path, seq_length, img_size, use_optical_flow):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = (list(range(total_frames)) if total_frames <= seq_length
               else np.linspace(0, total_frames - 1, seq_length, dtype=int).tolist())
    rgb_tf  = VideoNumpyTransform(img_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    freq_tf = VideoNumpyTransform(img_size, (0.5, 0.5, 0.5),       (0.5, 0.5, 0.5))
    rgb_list, freq_list = [], []
    prev_idx = -1
    for target_idx in indices:
        if target_idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame_bgr = cap.read()
        prev_idx = target_idx
        zero3 = (3, img_size, img_size)
        if not ret:
            rgb_list.append(torch.zeros(zero3)); freq_list.append(torch.zeros(zero3)); continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_list.append(rgb_tf(frame_rgb))
        freq_list.append(freq_tf(get_frequency_features(frame_rgb)))
    cap.release()
    zero3 = (3, img_size, img_size)
    while len(rgb_list) < seq_length:
        rgb_list.append(rgb_list[-1].clone()  if rgb_list  else torch.zeros(zero3))
        freq_list.append(freq_list[-1].clone() if freq_list else torch.zeros(zero3))
    video = torch.cat([torch.stack(rgb_list[:seq_length]), torch.stack(freq_list[:seq_length])], dim=1)
    return video.unsqueeze(0)


def predict_video(video_path, model, video_cfg, device):
    tensor = _video_to_tensor(video_path, video_cfg["seq_length"], video_cfg["img_size"], video_cfg["use_optical_flow"]).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()
    fake_pct   = round(prob * 100, 2)
    real_pct   = round((1 - prob) * 100, 2)
    confidence = max(prob, 1 - prob) * 100
    conf_level = ("ÇOK YÜKSEK" if confidence > 90 else "YÜKSEK" if confidence > 75 else "ORTA" if confidence > 60 else "DÜŞÜK")
    label = "SAHTE" if prob >= 0.5 else "GERÇEK"
    return {
        "fake_pct": fake_pct, "real_pct": real_pct, "label": label,
        "confidence": round(confidence, 2), "confidence_level": conf_level,
        "logit": round(logit.item(), 4), "seq_length": video_cfg["seq_length"],
        "img_size": video_cfg["img_size"],
        "val_acc": f"{video_cfg.get('val_acc'):.2f}%" if video_cfg.get("val_acc") else "—",
        "val_f1":  f"{video_cfg.get('val_f1'):.3f}"  if video_cfg.get("val_f1")  else "—",
        "val_auc": f"{video_cfg.get('val_auc'):.3f}"  if video_cfg.get("val_auc")  else "—",
        "scores": {"Sahte Olasılığı": fake_pct, "Gerçek Olasılığı": real_pct},
    }


def predict_image(image_path, model, image_cfg, device):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from PIL import Image as PILImage
    img_size  = image_cfg.get("image_size", 384)
    transform = A.Compose([A.Resize(img_size, img_size),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])
    img_np    = np.array(PILImage.open(image_path).convert("RGB"))
    tensor    = transform(image=img_np)["image"].unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()
    real_pct   = round(prob * 100, 2)
    fake_pct   = round((1 - prob) * 100, 2)
    confidence = max(prob, 1 - prob) * 100
    conf_level = ("ÇOK YÜKSEK" if confidence > 90 else "YÜKSEK" if confidence > 75 else "ORTA" if confidence > 60 else "DÜŞÜK")
    label = "GERÇEK" if prob >= 0.5 else "SAHTE"
    return {
        "fake_pct": fake_pct, "real_pct": real_pct, "label": label,
        "confidence": round(confidence, 2), "confidence_level": conf_level,
        "logit": round(logit.item(), 4), "image_size": img_size,
        "val_acc":  f"{image_cfg.get('val_acc'):.2f}%"  if image_cfg.get("val_acc")  else "—",
        "val_f1":   f"{image_cfg.get('val_f1'):.3f}"    if image_cfg.get("val_f1")   else "—",
        "val_auc":  f"{image_cfg.get('val_auc'):.3f}"   if image_cfg.get("val_auc")  else "—",
        "val_precision": f"{image_cfg.get('val_precision'):.3f}" if image_cfg.get("val_precision") else "—",
        "val_recall":    f"{image_cfg.get('val_recall'):.3f}"    if image_cfg.get("val_recall")    else "—",
        "scores": {"Sahte Olasılığı": fake_pct, "Gerçek Olasılığı": real_pct},
    }


def predict_text(text, model, tokenizer, scaler, config, device):
    raw_feats    = _extract_style_features(text)
    scaled_feats = scaler.transform(raw_feats)
    style_tensor = torch.tensor(scaled_feats, dtype=torch.float32).to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH, padding=True)
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, style_tensor, token_type_ids)
        probs  = torch.softmax(logits, dim=1)[0]
    confidence_ai = probs[1].item()
    confidence_hu = probs[0].item()
    if confidence_ai > config.AI_THRESHOLD:
        label = "SAHTE"; skor = confidence_ai
    else:
        label = "GERÇEK"; skor = confidence_hu
    fake_pct   = round(confidence_ai * 100, 2)
    real_pct   = round(confidence_hu * 100, 2)
    confidence = round(skor * 100, 2)
    conf_level = ("ÇOK YÜKSEK" if confidence > 90 else "YÜKSEK" if confidence > 75 else "ORTA" if confidence > 60 else "DÜŞÜK")
    tokens        = tokenizer.tokenize(text)
    token_count   = len(tokens)
    unique_tokens = len(set(tokens))
    lex_div       = round(unique_tokens / token_count * 100, 1) if token_count > 0 else 0.0
    return {
        "fake_pct": fake_pct, "real_pct": real_pct, "label": label,
        "confidence": confidence, "confidence_level": conf_level,
        "word_count": len(text.split()), "char_count": len(text),
        "token_count": token_count, "unique_tokens": unique_tokens,
        "lexical_diversity": lex_div,
        "scores": {"Sahte Olasılığı": fake_pct, "Gerçek Olasılığı": real_pct},
    }


def _extract_text_from_uploaded(uploaded_file) -> str:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    text = ""
    try:
        if ext in ("txt", "md"):
            raw = uploaded_file.read()
            try:    text = raw.decode("utf-8")
            except: text = raw.decode("latin-1", errors="replace")
        elif ext == "pdf":
            import pdfplumber
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                tf.write(uploaded_file.read()); tf_path = tf.name
            with pdfplumber.open(tf_path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
            os.unlink(tf_path)
        elif ext == "docx":
            import docx as _docx
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tf:
                tf.write(uploaded_file.read()); tf_path = tf.name
            doc  = _docx.Document(tf_path)
            text = "\n".join(p.text for p in doc.paragraphs).strip()
            os.unlink(tf_path)
    except Exception as e:
        st.error(f"❌ Dosya okunamadı: {e}")
    return text


def run_audio_analysis(uploaded_file, model, config, device):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read()); tmp_path = tmp.name
    try:    result = predict_audio(tmp_path, model, config, device)
    finally: os.unlink(tmp_path)
    return {"ai_pct": result["fake_pct"], "scores": result["scores"],
            "file": uploaded_file.name, "mode": "🎵 Ses", "detail": result, "real": True}


def run_video_analysis(uploaded_file, model, video_cfg, device):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read()); tmp_path = tmp.name
    try:    result = predict_video(tmp_path, model, video_cfg, device)
    finally: os.unlink(tmp_path)
    return {"ai_pct": result["fake_pct"], "scores": result["scores"],
            "file": uploaded_file.name, "mode": "🎬 Video", "detail": result, "real": True}


def run_image_analysis(uploaded_file, model, image_cfg, device):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read()); tmp_path = tmp.name
    try:    result = predict_image(tmp_path, model, image_cfg, device)
    finally: os.unlink(tmp_path)
    return {"ai_pct": result["fake_pct"], "scores": result["scores"],
            "file": uploaded_file.name, "mode": "🖼️ Görsel", "detail": result, "real": True}


def run_text_analysis(text, model, tokenizer, scaler, config, device, filename=None):
    result = predict_text(text, model, tokenizer, scaler, config, device)
    return {"ai_pct": result["fake_pct"], "scores": result["scores"],
            "file": filename or f"metin_{len(text)}karakter",
            "mode": "📝 Metin", "detail": result, "real": True}


# ══════════════════════════════════════════════════════════════════════════════
#  HTML BİLEŞENLERİ
# ══════════════════════════════════════════════════════════════════════════════

def donut_chart(ai_pct, label="Yapay Zeka Üretimi"):
    c = 2 * 3.14159 * 80
    ai_d, re_d = c * ai_pct / 100, c * (100 - ai_pct) / 100
    if ai_pct >= 70:
        col, vc = "#EF4444", "#EF4444"
        vbg, vb, vt = "rgba(239,68,68,0.12)", "rgba(239,68,68,0.3)", "⚠️ YÜKSEK OLASILIKLA YAPAY ZEKA"
    elif ai_pct >= 40:
        col, vc = "#F59E0B", "#F59E0B"
        vbg, vb, vt = "rgba(245,158,11,0.12)", "rgba(245,158,11,0.3)", "⚡ KESİN DEĞİL — DEĞERLENDİRME GEREKLİ"
    else:
        col, vc = "#06B6D4", "#06B6D4"
        vbg, vb, vt = "rgba(6,182,212,0.12)", "rgba(6,182,212,0.3)", "✅ GERÇEK"
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:1.5rem;">
      <div style="position:relative;width:200px;height:200px;display:flex;align-items:center;justify-content:center;">
        <svg width="200" height="200" viewBox="0 0 200 200" style="transform:rotate(-90deg);">
          <circle cx="100" cy="100" r="80" fill="none" stroke="#353439" stroke-width="16"/>
          <circle cx="100" cy="100" r="80" fill="none" stroke="{col}" stroke-width="16"
            stroke-dasharray="{ai_d:.1f} {c:.1f}" stroke-linecap="round"/>
          <circle cx="100" cy="100" r="80" fill="none" stroke="#7C3AED" stroke-width="16"
            stroke-dasharray="{re_d:.1f} {c:.1f}" stroke-dashoffset="{-ai_d:.1f}"
            stroke-linecap="round" opacity="0.4"/>
        </svg>
        <div style="position:absolute;text-align:center;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:800;color:#e4e1e7;">{ai_pct:.0f}%</div>
          <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;color:#ccc3d8;">{label}</div>
        </div>
      </div>
      <div style="width:100%;padding:0.85rem 1.2rem;background:{vbg};border:1px solid {vb};
                  border-radius:12px;text-align:center;font-family:'Space Grotesk',sans-serif;
                  font-weight:700;color:{vc};font-size:0.85rem;letter-spacing:0.05em;">{vt}</div>
    </div>"""


def score_bar(label, value, color="#06B6D4"):
    return f"""
    <div style="margin-bottom:1rem;">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.82rem;">
        <span style="color:#e4e1e7;font-weight:500;">{label}</span>
        <span style="color:{color};font-weight:700;">{value:.0f}%</span>
      </div>
      <div style="height:6px;background:#353439;border-radius:4px;overflow:hidden;">
        <div style="height:100%;width:{value}%;background:linear-gradient(90deg,#7C3AED,{color});border-radius:4px;"></div>
      </div>
    </div>"""


def stat_card(title, value, border_color="#7C3AED"):
    return f"""
    <div style="background:#1b1b1f;border-radius:14px;padding:1rem 1.25rem;border-left:3px solid {border_color};">
      <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#ccc3d8;margin-bottom:4px;">{title}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;color:#e4e1e7;">{value}</div>
    </div>"""


def history_row(filename, mode, ts, ai_pct):
    if ai_pct >= 70:
        badge = '<span style="padding:2px 10px;border-radius:20px;font-size:0.65rem;font-weight:700;background:rgba(255,180,171,0.12);color:#ffb4ab;border:1px solid rgba(255,180,171,0.25);">YAPAY ZEKA TESPİT EDİLDİ</span>'
        sc = "#ffb4ab"
    elif ai_pct >= 40:
        badge = '<span style="padding:2px 10px;border-radius:20px;font-size:0.65rem;font-weight:700;background:rgba(245,158,11,0.12);color:#F59E0B;border:1px solid rgba(245,158,11,0.25);">ŞÜPHELİ</span>'
        sc = "#F59E0B"
    else:
        badge = '<span style="padding:2px 10px;border-radius:20px;font-size:0.65rem;font-weight:700;background:rgba(6,182,212,0.12);color:#06B6D4;border:1px solid rgba(6,182,212,0.25);">ÖZGÜN</span>'
        sc = "#06B6D4"
    icon = {"🎵 Ses":"🎵","🖼️ Görsel":"🖼️","📝 Metin":"📝","🎬 Video":"🎬"}.get(mode,"📄")
    return f"""
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
      <td style="padding:12px 16px;font-size:0.82rem;color:#e4e1e7;">{icon} {filename}</td>
      <td style="padding:12px 16px;font-size:0.75rem;color:#ccc3d8;">{mode}</td>
      <td style="padding:12px 16px;font-size:0.75rem;color:#ccc3d8;">{ts}</td>
      <td style="padding:12px 16px;">{badge}</td>
      <td style="padding:12px 16px;font-family:'Space Grotesk',sans-serif;font-size:0.85rem;font-weight:700;color:{sc};">{ai_pct:.0f}% YZ</td>
    </tr>"""


def model_status_row(icon, label, ready, model_name):
    color = "#06B6D4" if ready else "#4a4455"
    dot   = "●" if ready else "○"
    return f"""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="font-size:1rem;">{icon}</span>
      <div style="flex:1;min-width:0;">
        <div style="font-size:0.78rem;font-weight:600;color:#e4e1e7;">{label}</div>
        <div style="font-size:0.62rem;color:#4a4455;white-space:nowrap;overflow:hidden;
                    text-overflow:ellipsis;">{model_name}</div>
      </div>
      <span style="color:{color};font-size:0.7rem;">{dot}</span>
    </div>"""


# ── Görsel önizleme için yardımcı fonksiyon ───────────────────────────────
def _render_image_preview(img_bytes: bytes, filename: str):


    ext  = filename.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    b64  = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""<div class="aibuster-img-wrapper">
  <img src="data:{mime};base64,{b64}" class="aibuster-img-preview" alt="{filename}" />
</div>
<div style="font-size:0.65rem;color:#4a4455;margin-top:4px;max-width:320px;">{filename}</div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "uploaded_image_id" not in st.session_state:
    st.session_state.uploaded_image_id    = None
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "detected_mode" not in st.session_state:
    st.session_state.detected_mode = None

audio_model, audio_config, audio_device, audio_err                           = get_audio_model()
video_model, video_cfg,    video_device, video_err                            = get_video_model()
image_model, image_cfg,    image_device, image_err                            = get_image_model()
text_model,  text_tokenizer, text_scaler, text_config, text_device, text_err = get_text_model()

audio_ready = audio_model is not None
video_ready = video_model is not None
image_ready = image_model is not None
text_ready  = text_model  is not None and text_scaler is not None


# ══════════════════════════════════════════════════════════════════════════════
#  KENAR ÇUBUĞU
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0 0 1.5rem 0;">
      <div style="display:flex;align-items:center;gap:12px;">
        {logo_img_tag(42)}
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
                      font-weight:800;color:#7C3AED;line-height:1.1;">AI-BUSTER</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);
                border-radius:12px;padding:12px 14px;margin-bottom:1.2rem;">
      <div style="font-size:0.72rem;font-weight:700;color:#7C3AED;margin-bottom:6px;
                  text-transform:uppercase;letter-spacing:0.08em;">🤖 Akıllı Tespit</div>
      <div style="font-size:0.7rem;color:#ccc3d8;line-height:1.5;">
        Dosya uzantısına göre uygun model <strong style="color:#e4e1e7;">otomatik seçilir</strong>.
        Ayrıca metin girerek doğrudan analiz yapabilirsiniz.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:1.2rem;">
      <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                  color:#4a4455;margin-bottom:8px;font-weight:600;">Desteklenen Formatlar</div>
      <div style="display:flex;flex-wrap:wrap;gap:4px;">
        <span style="padding:2px 8px;background:#1b1b1f;border:1px solid rgba(255,255,255,0.08);
                     border-radius:6px;font-size:0.62rem;color:#ccc3d8;">🎵 Ses</span>
        <span style="padding:2px 8px;background:#1b1b1f;border:1px solid rgba(255,255,255,0.08);
                     border-radius:6px;font-size:0.62rem;color:#ccc3d8;">🖼️ Görsel</span>
        <span style="padding:2px 8px;background:#1b1b1f;border:1px solid rgba(255,255,255,0.08);
                     border-radius:6px;font-size:0.62rem;color:#ccc3d8;">🎬 Video</span>
        <span style="padding:2px 8px;background:#1b1b1f;border:1px solid rgba(255,255,255,0.08);
                     border-radius:6px;font-size:0.62rem;color:#ccc3d8;">📝 Metin</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0 0 1rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a4455;margin-bottom:8px;font-weight:600;">Model Durumları</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#1b1b1f;border-radius:12px;padding:8px 12px;">
      {model_status_row("🎵", "Ses Dedektörü",    audio_ready, "Wav2Vec2 + LFCC-LCNN")}
      {model_status_row("🖼️", "Görsel Dedektörü", image_ready, "SigLIP2-so400m-patch14")}
      {model_status_row("🎬", "Video Dedektörü",  video_ready, "EfficientNet-B5 + Transformer")}
      {model_status_row("📝", "Metin Dedektörü",  text_ready,  "BERT Türkçe (dbmdz)")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:0.75rem;">
      {stat_card("Toplam Analiz",  f"{st.session_state.total_analyzed:,}")}
      {stat_card("Deepfake Tespiti", f"{st.session_state.deepfake_hits:,}", "#06B6D4")}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANA BAŞLIK
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:2rem;">
  <h1 style="font-size:2.4rem;font-weight:800;margin:0;line-height:1.1;">
    AI <span style="color:#06B6D4;">BUSTER</span>
  </h1>
  <p style="color:#ccc3d8;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.12em;margin-top:6px;">
    Sistem Durumu: Ortam Taranıyor · Dosya uzantısına göre otomatik model seçimi
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANA İÇERİK
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([6, 4], gap="large")

with col_left:
    st.markdown("""
    <div style="background:#1b1b1f;border-radius:20px;padding:2rem;margin-bottom:1.5rem;
                position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,transparent,#06B6D4,transparent);opacity:0.4;"></div>
      <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:700;margin:0 0 0.5rem 0;">
        📂 Dosya Yükle veya ✏️ Metin Gir
      </h3>
      <p style="font-size:0.78rem;color:#ccc3d8;margin:0;">
        Model, yüklediğiniz dosyanın türüne göre <strong style="color:#e4e1e7;">otomatik seçilir</strong>.
        Ses, görsel, video veya metin dosyası yükleyebilirsiniz.
      </p>
    </div>""", unsafe_allow_html=True)

    tab_file, tab_manual = st.tabs(["📂 Dosya Yükle", "✏️ Metin Gir"])

    # ─── SEKMİ 1: Dosya yükleme ───────────────────────────────────────────
    with tab_file:
        uploaded = st.file_uploader(
            "Dosyanızı buraya sürükleyin",
            type=ALL_EXTS,
            label_visibility="collapsed",
            key="unified_uploader",
        )
        detected_mode = None
        if uploaded:
            detected_mode = detect_mode_from_ext(uploaded.name)
            mode_icons = {"🎵 Ses": "#06B6D4", "🖼️ Görsel": "#7C3AED", "🎬 Video": "#F59E0B", "📝 Metin": "#06B6D4"}
            mc = mode_icons.get(detected_mode, "#7C3AED")
            st.markdown(f"""
            <div style="margin:0.75rem 0;padding:8px 14px;
                        background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.25);
                        border-radius:10px;display:inline-flex;align-items:center;gap:8px;">
              <span style="font-size:0.68rem;color:#ccc3d8;">Tespit edilen tür:</span>
              <span style="font-size:0.75rem;font-weight:700;color:{mc};">{detected_mode}</span>
              <span style="font-size:0.65rem;color:#4a4455;">→ model otomatik seçildi</span>
            </div>
            """, unsafe_allow_html=True)

            if detected_mode == "🎵 Ses":
                st.audio(uploaded, format=f"audio/{uploaded.name.rsplit('.',1)[-1]}")
                uploaded.seek(0)
                st.markdown(
                    '<div style="margin-top:6px;padding:6px 12px;background:rgba(6,182,212,0.07);'
                    'border-left:3px solid #06B6D4;border-radius:6px;font-size:0.7rem;color:#ccc3d8;">'
                    '💡 <strong style="color:#e4e1e7;">İpucu:</strong> Daha doğru sonuç için '
                    '<strong style="color:#06B6D4;">en az 20 saniye</strong> uzunluğunda ses kaydı önerilir.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            elif detected_mode == "🎬 Video":
                st.video(uploaded)
                uploaded.seek(0)

            elif detected_mode == "🖼️ Görsel":
                # ── Titreme düzeltmesi: st.columns + st.image YOK ──
                # Görsel bytes'ı bir kez session_state'e kaydedilir,
                # sonraki render'larda aynı bytes kullanılır → layout değişmez.
                file_id = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.uploaded_image_id != file_id:
                    uploaded.seek(0)
                    st.session_state.uploaded_image_bytes = uploaded.read()
                    st.session_state.uploaded_image_id   = file_id
                    uploaded.seek(0)
                if st.session_state.uploaded_image_bytes:
                    _render_image_preview(
                        st.session_state.uploaded_image_bytes,
                        uploaded.name,
                    )

            elif detected_mode == "📝 Metin":
                try:
                    uploaded.seek(0)
                    preview_text = _extract_text_from_uploaded(uploaded)
                    uploaded.seek(0)
                    if preview_text:
                        preview_short = preview_text[:400] + ("…" if len(preview_text) > 400 else "")
                        wc = len(preview_text.split())
                        cc = len(preview_text)
                        st.markdown(
                            f'<div style="background:#131317;border:1px solid rgba(255,255,255,0.07);'
                            f'border-radius:10px;padding:1rem;font-size:0.78rem;color:#ccc3d8;'
                            f'line-height:1.6;max-height:160px;overflow-y:auto;">{preview_short}</div>'
                            f'<div style="font-size:0.7rem;color:#4a4455;text-align:right;margin-top:4px;">'
                            f'{wc} kelime · {cc} karakter</div>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        analyze_file_btn = st.button(
            "🔍 Analizi Başlat",
            disabled=(uploaded is None),
            use_container_width=True,
            key="analyze_file_btn",
        )

    # ─── SEKMİ 2: Manuel metin girişi ─────────────────────────────────────
    with tab_manual:
        text_input_value_raw = st.text_area(
            label="Analiz edilecek metin",
            placeholder=(
                "Buraya analiz etmek istediğiniz metni yazın veya yapıştırın…\n\n"
                f"Not: Analiz için en az {MIN_WORDS_REQUIRED} kelime gereklidir.\n\n"
                "Örnek: Makale, haber metni, e-posta, sosyal medya gönderisi vb."
            ),
            height=220,
            label_visibility="collapsed",
            key="text_input_area",
        )

        word_count = len(text_input_value_raw.split()) if text_input_value_raw.strip() else 0
        char_count = len(text_input_value_raw)
        words_ok   = word_count >= MIN_WORDS_REQUIRED
        wc_color   = "#06B6D4" if words_ok else "#F59E0B"

        if word_count > 0:
            remaining_w = max(0, MIN_WORDS_REQUIRED - word_count)
            hint_parts  = f'<span style="color:{wc_color};font-weight:600;">{word_count}/{MIN_WORDS_REQUIRED} kelime</span>'
            if remaining_w > 0:
                hint_parts += f' <span style="color:#F59E0B;">· {remaining_w} kelime daha girin</span>'
            else:
                hint_parts += ' <span style="color:#06B6D4;">✓</span>'
            hint_parts += f' &nbsp;·&nbsp; <span style="color:#4a4455;">{char_count} karakter</span>'
        else:
            hint_parts = f'<span style="color:#4a4455;">0/{MIN_WORDS_REQUIRED} kelime</span>'

        st.markdown(
            f'<div style="font-size:0.7rem;text-align:right;margin-top:4px;">{hint_parts}</div>',
            unsafe_allow_html=True,
        )

        text_input_value = text_input_value_raw if (text_input_value_raw.strip() and words_ok) else ""

        analyze_text_btn = st.button(
            "🔍 Metni Analiz Et",
            disabled=(not text_input_value_raw.strip() or not words_ok),
            use_container_width=True,
            key="analyze_text_btn",
        )

        if text_input_value_raw.strip() and not words_ok:
            remaining = MIN_WORDS_REQUIRED - word_count
            st.warning(f"⚠️ Analiz için en az **{MIN_WORDS_REQUIRED} kelime** gereklidir. {remaining} kelime daha girin.")

    # ══════════════════════════════════════════════════════════════════════
    #  ANALİZ İŞLEYİCİSİ
    # ══════════════════════════════════════════════════════════════════════

    result = None

    if analyze_file_btn and uploaded:
        mode = detect_mode_from_ext(uploaded.name)
        if mode is None:
            st.error("❌ Desteklenmeyen dosya formatı.")

        elif mode == "🎵 Ses":
            if audio_ready:
                with st.spinner("🎵 Ses modeli çalışıyor, lütfen bekleyin…"):
                    try:
                        uploaded.seek(0)
                        result = run_audio_analysis(uploaded, audio_model, audio_config, audio_device)
                    except Exception as e:
                        st.error(f"❌ Analiz hatası: {e}")
            else:
                st.error("❌ Ses modeli yüklenemedi. Lütfen daha sonra tekrar deneyin.")

        elif mode == "🖼️ Görsel":
            if image_ready:
                with st.spinner("🖼️ Görsel modeli çalışıyor, lütfen bekleyin…"):
                    try:
                        uploaded.seek(0)
                        result = run_image_analysis(uploaded, image_model, image_cfg, image_device)
                    except Exception as e:
                        st.error(f"❌ Görsel analiz hatası: {e}")
            else:
                st.error("❌ Görsel modeli yüklenemedi. Lütfen daha sonra tekrar deneyin.")

        elif mode == "🎬 Video":
            if video_ready:
                with st.spinner("🎬 Video modeli çalışıyor, lütfen bekleyin…"):
                    try:
                        uploaded.seek(0)
                        result = run_video_analysis(uploaded, video_model, video_cfg, video_device)
                    except Exception as e:
                        st.error(f"❌ Video analiz hatası: {e}")
            else:
                st.error("❌ Video modeli yüklenemedi. Lütfen daha sonra tekrar deneyin.")

        elif mode == "📝 Metin":
            uploaded.seek(0)
            file_text = _extract_text_from_uploaded(uploaded)
            uploaded.seek(0)
            if file_text.strip():
                file_word_count = len(file_text.split())
                if file_word_count < MIN_WORDS_REQUIRED:
                    st.warning(f"⚠️ Dosyadaki metin çok kısa ({file_word_count} kelime). En az {MIN_WORDS_REQUIRED} kelime gereklidir.")
                elif text_ready:
                    with st.spinner("📝 Metin modeli çalışıyor, lütfen bekleyin…"):
                        try:
                            result = run_text_analysis(file_text, text_model, text_tokenizer, text_scaler, text_config, text_device, filename=uploaded.name)
                        except Exception as e:
                            st.error(f"❌ Metin analiz hatası: {e}")
                else:
                    st.error("❌ Metin modeli yüklenemedi. Lütfen daha sonra tekrar deneyin.")
            else:
                st.warning("⚠️ Metin dosyasından içerik çıkarılamadı.")

    if analyze_text_btn:
        if not text_input_value.strip():
            st.warning("⚠️ Lütfen analiz edilecek bir metin girin.")
        elif text_ready:
            with st.spinner("📝 Metin modeli çalışıyor, lütfen bekleyin…"):
                try:
                    result = run_text_analysis(text_input_value, text_model, text_tokenizer, text_scaler, text_config, text_device)
                except Exception as e:
                    st.error(f"❌ Metin analiz hatası: {e}")
        else:
            st.error("❌ Metin modeli yüklenemedi. Lütfen daha sonra tekrar deneyin.")

    if result:
        st.session_state.last_result = result
        new_entry   = {"file": result["file"], "mode": result["mode"], "ts": datetime.now().strftime("%H:%M"), "ai": result["ai_pct"]}
        is_deepfake = result["ai_pct"] >= 70
        st.session_state.history.insert(0, new_entry)
        st.session_state.history        = st.session_state.history[:500]
        st.session_state.total_analyzed += 1
        if is_deepfake:
            st.session_state.deepfake_hits += 1
        save_analytics(new_entry, is_deepfake)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SAĞ SÜTUN: SONUÇ PANELİ
# ══════════════════════════════════════════════════════════════════════════════
with col_right:
    result = st.session_state.last_result
    if result:
        ai_pct        = result["ai_pct"]
        is_real_model = result.get("real", False)
        mode          = result.get("mode", "")
        mode_is_audio = mode == "🎵 Ses"
        mode_is_video = mode == "🎬 Video"
        mode_is_image = mode == "🖼️ Görsel"
        mode_is_text  = mode == "📝 Metin"

        model_badge_map = {
            "🎵 Ses":    "Ses Modeli Aktif",
            "🖼️ Görsel": "Görsel Modeli Aktif",
            "🎬 Video":  "Video Modeli Aktif",
            "📝 Metin":  "Metin Modeli Aktif",
        }
        if is_real_model:
            badge_html = f'<span style="font-size:0.6rem;font-weight:700;padding:3px 10px;border-radius:20px;background:rgba(6,182,212,0.12);color:#06B6D4;border:1px solid rgba(6,182,212,0.25);text-transform:uppercase;letter-spacing:0.1em;">{model_badge_map.get(mode, "Aktif")}</span>'
        else:
            badge_html = '<span style="font-size:0.6rem;font-weight:700;padding:3px 10px;border-radius:20px;background:rgba(124,58,237,0.12);color:#7C3AED;border:1px solid rgba(124,58,237,0.25);text-transform:uppercase;letter-spacing:0.1em;">Demo Analizi</span>'

        st.markdown(f"""
        <div style="background:#1b1b1f;border-radius:20px;padding:2rem;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#7C3AED,transparent);opacity:0.5;"></div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
            <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;margin:0;">Tespit Sonuçları</h2>
            {badge_html}
          </div>
          <div style="font-size:0.68rem;color:#4a4455;margin-bottom:1.5rem;">{mode}</div>
        """, unsafe_allow_html=True)

        label_str = "Sahte Olasılığı" if is_real_model else "Yapay Zeka Üretimi"
        st.markdown(donut_chart(ai_pct, label_str), unsafe_allow_html=True)
        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

        if result.get("scores"):
            st.markdown("".join(score_bar(k, v) for k, v in result["scores"].items()), unsafe_allow_html=True)

        detail_modes = [
            (mode_is_audio, "🎵 Ses"),
            (mode_is_image, "🖼️ Görsel"),
            (mode_is_video, "🎬 Video"),
            (mode_is_text,  "📝 Metin"),
        ]
        for is_this_mode, _ in detail_modes:
            if is_real_model and is_this_mode and "detail" in result:
                d = result["detail"]
                lc = "#ffb4ab" if d["label"] == "SAHTE" else "#06B6D4"
                st.markdown(f"""
                <div style="margin-top:1.5rem;padding:1rem;background:#131317;border-radius:12px;border:1px solid rgba(255,255,255,0.06);">
                  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#ccc3d8;margin-bottom:0.75rem;font-weight:600;">Model Detayları</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.78rem;">
                    <div style="color:#ccc3d8;">Karar</div>
                    <div style="color:{lc};font-weight:700;">{d['label']}</div>
                    <div style="color:#ccc3d8;">Güven Seviyesi</div>
                    <div style="color:#e4e1e7;font-weight:600;">{d['confidence_level']} ({d['confidence']:.1f}%)</div>
                  </div>
                </div>""", unsafe_allow_html=True)
                break

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#1b1b1f;border-radius:20px;padding:3rem 2rem;text-align:center;border:2px dashed rgba(124,58,237,0.2);">
          <div style="font-size:3rem;margin-bottom:1rem;">🛡️</div>
          <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#e4e1e7;margin:0 0 0.5rem 20px;">Analiz Bekleniyor</h3>
          <p style="font-size:0.8rem;color:#ccc3d8;margin:0 0 1rem 0;">
            Bir dosya yükleyin ya da metin girin — model otomatik seçilir.
          </p>
          <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;font-size:0.72rem;color:#4a4455;">
            <span>🎵 Ses</span><span>🖼️ Görsel</span><span>🎬 Video</span><span>📝 Metin</span>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SON 5 ANALİZ KAYDI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)

son_5 = st.session_state.history[:5]
if son_5:
    rows_html = "".join(history_row(h["file"], h["mode"], h["ts"], h["ai"]) for h in son_5)
else:
    rows_html = '<tr><td colspan="5" style="padding:2rem;text-align:center;color:#4a4455;font-size:0.82rem;">Henüz analiz yapılmadı</td></tr>'

st.markdown(f"""
<div style="background:#1b1b1f;border-radius:20px;overflow:hidden;position:relative;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#7C3AED,transparent);opacity:0.4;"></div>
  <div style="padding:1.4rem 1.75rem;border-bottom:1px solid rgba(255,255,255,0.05);display:flex;align-items:center;justify-content:space-between;">
    <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:700;margin:0;">Son Analiz Kayıtları</h2>
    <span style="font-size:0.65rem;color:#4a4455;font-weight:500;">Son {len(son_5)} kayıt gösteriliyor</span>
  </div>
  <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
      <thead>
        <tr style="background:#0e0e12;">
          <th style="padding:10px 16px;text-align:left;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#ccc3d8;font-weight:600;">Kaynak Dosya</th>
          <th style="padding:10px 16px;text-align:left;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#ccc3d8;font-weight:600;">Mod</th>
          <th style="padding:10px 16px;text-align:left;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#ccc3d8;font-weight:600;">Saat</th>
          <th style="padding:10px 16px;text-align:left;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#ccc3d8;font-weight:600;">Durum</th>
          <th style="padding:10px 16px;text-align:left;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#ccc3d8;font-weight:600;">Puan</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
<div style='height:2rem;'></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  GERİ BİLDİRİM MEKANİZMASI
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_feedback_from_hf() -> list:
    try:
        path = hf_hub_download(
            repo_id=HF_ANALYTICS_REPO, filename=HF_FEEDBACK_FILE,
            repo_type="dataset", token=HF_TOKEN_W, force_download=True,
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_feedback(entry: dict):
    existing = _fetch_feedback_from_hf()
    existing.insert(0, entry)
    existing = existing[:1000]
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=json.dumps(existing, ensure_ascii=False, indent=2).encode("utf-8"),
            path_in_repo=HF_FEEDBACK_FILE,
            repo_id=HF_ANALYTICS_REPO,
            repo_type="dataset",
            token=HF_TOKEN_W,
            commit_message=f"Geri bildirim eklendi: {entry.get('file', '?')}",
        )
        return True
    except Exception as e:
        st.warning(f"⚠️ Geri bildirim kaydedilemedi: {e}")
        return False


if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "feedback_last_result_file" not in st.session_state:
    st.session_state.feedback_last_result_file = None

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

last = st.session_state.last_result

if last and last.get("file") != st.session_state.feedback_last_result_file:
    st.session_state.feedback_submitted = False
    st.session_state.feedback_last_result_file = last.get("file")

st.markdown("""
<div style="background:#1b1b1f;border-radius:20px;overflow:hidden;position:relative;margin-bottom:2rem;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;
              background:linear-gradient(90deg,transparent,#06B6D4,transparent);opacity:0.4;"></div>
  <div style="padding:1.4rem 1.75rem;border-bottom:1px solid rgba(255,255,255,0.05);">
    <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:700;margin:0 0 4px 0;">
      💬 Geri Bildirim
    </h2>
    <p style="font-size:0.72rem;color:#ccc3d8;margin:0;">
      Modelin tahmini doğru muydu? Deneyiminizi paylaşarak sistemi geliştirmemize yardımcı olun.
    </p>
  </div>
""", unsafe_allow_html=True)

if not last:
    st.markdown("""
    <div style="padding:2rem 1.75rem;text-align:center;">
      <p style="font-size:0.8rem;color:#4a4455;margin:0;">Geri bildirim vermek için önce bir analiz yapın.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.feedback_submitted:
    st.markdown("""
    <div style="padding:2rem 1.75rem;text-align:center;">
      <div style="font-size:2rem;margin-bottom:0.75rem;">✅</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:#06B6D4;margin-bottom:0.4rem;">Teşekkürler!</div>
      <p style="font-size:0.78rem;color:#ccc3d8;margin:0;">Geri bildiriminiz kaydedildi. Sistemi geliştirmemize yardımcı oluyorsunuz.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("</div>", unsafe_allow_html=True)

    ai_pct      = last.get("ai_pct", 0)
    model_label = "SAHTE" if ai_pct >= 50 else "GERÇEK"
    label_color = "#ffb4ab" if model_label == "SAHTE" else "#06B6D4"

    st.markdown(f"""
    <div style="padding:1.25rem 1.75rem;background:#131317;border-bottom:1px solid rgba(255,255,255,0.05);">
      <div style="font-size:0.68rem;color:#ccc3d8;margin-bottom:4px;">
        Son analiz sonucu: <strong style="color:{label_color};">{model_label}</strong>
        <span style="color:#4a4455;"> ({ai_pct:.1f}% YZ)</span>
      </div>
      <div style="font-size:0.65rem;color:#4a4455;">{last.get('file', '—')} · {last.get('mode', '—')}</div>
    </div>
    """, unsafe_allow_html=True)

    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 1], gap="small")

    with fb_col1:
        st.markdown('<div style="padding:1rem 1.25rem;"><div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a4455;margin-bottom:8px;font-weight:600;">Tahmin doğru muydu?</div></div>', unsafe_allow_html=True)
        verdict = st.radio("Tahmin", ["✅ Evet, doğru", "❌ Hayır, yanlış", "🤷 Emin değilim"], label_visibility="collapsed", key="fb_verdict")

    with fb_col2:
        st.markdown('<div style="padding:1rem 1.25rem;"><div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a4455;margin-bottom:8px;font-weight:600;">Güven skoru nasıldı?</div></div>', unsafe_allow_html=True)
        confidence_rating = st.radio("Güven", ["🎯 Çok uygun", "📊 Makul", "⚠️ Çok düşük"], label_visibility="collapsed", key="fb_confidence")

    with fb_col3:
        st.markdown('<div style="padding:1rem 1.25rem;"><div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a4455;margin-bottom:8px;font-weight:600;">Genel deneyim</div></div>', unsafe_allow_html=True)
        experience = st.radio("Deneyim", ["⭐⭐⭐ Mükemmel", "⭐⭐ İyi", "⭐ Geliştirilmeli"], label_visibility="collapsed", key="fb_experience")

    st.markdown("<div style='padding:0 1.25rem;'>", unsafe_allow_html=True)
    comment = st.text_area("Ek yorum (isteğe bağlı)", placeholder="Modelin neden yanlış/doğru olduğunu düşündüğünüzü yazabilirsiniz…", height=80, label_visibility="visible", key="fb_comment")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 1.25rem 1.25rem 1.25rem;'>", unsafe_allow_html=True)
    submit_fb = st.button("📨 Geri Bildirimi Gönder", use_container_width=True, key="submit_feedback_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    if submit_fb:
        fb_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": last.get("file", "—"), "mode": last.get("mode", "—"),
            "ai_pct": round(ai_pct, 2), "model_label": model_label,
            "verdict": verdict, "confidence_rating": confidence_rating,
            "experience": experience,
            "comment": comment.strip() if comment else "",
        }
        with st.spinner("📨 Geri bildirim kaydediliyor…"):
            ok = save_feedback(fb_entry)
        if ok:
            st.session_state.feedback_submitted = True
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


st.markdown("""
<div style="text-align: center; margin-top: 3rem; margin-bottom: 1rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05);">
    <p style="font-size: 0.75rem; color: #ccc3d8; margin: 0;">
        ⚠️ <b>Uyarı:</b> Bu araç yardımcı bir tespit sistemidir. Çıktılar kesin hukuki ya da adli kanıt niteliği taşımaz. Kritik kararlar için uzman değerlendirmesi önerilir.
    </p>
</div>
""", unsafe_allow_html=True)