<div align="center">

<img src="logo.jpg" width="90" style="border-radius:16px;" />

# AI-BUSTER

**Multi-Modal AI Content Detection System**

🇹🇷 [Türkçe](README_TR.md) · 🇬🇧 English

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-AI--BUSTER%20Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/AI-BUSTER/AI-BUSTER)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20Models-AI--BUSTER__Models-FFD21E?style=for-the-badge)](https://huggingface.co/AI-BUSTER/AI-BUSTER_Models)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

*Audio · Image · Video · Text — real-time deepfake detection in four modes, one interface*

</div>

---

## ✨ What's New?

> This release covers the project's **Gradio → Streamlit** migration, four independent neural network models, and a live demo running on Hugging Face Spaces.

| Feature | Old (Gradio) | New (Streamlit) |
|---|---|---|
| Interface | Gradio Blocks | Streamlit + custom CSS theme |
| Model loading | Local file | HuggingFace Hub (`hf_hub_download`) |
| Models | Audio + Image + Video | **Audio + Image + Video + Text** |
| Analytics | None | Persistent logging on HF Dataset |
| Demo | — | [🤗 HF Spaces](https://huggingface.co/spaces/AI-BUSTER/AI-BUSTER) |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Interface                │
│         (Upload file  ·  Enter text)                │
└────────────────────┬────────────────────────────────┘
                     │  auto-routing based on extension
        ┌────────────┼────────────┬────────────────┐
        ▼            ▼            ▼                ▼
   🎵 Audio      🖼 Image     🎬 Video         📝 Text
     Model          Model       Model            Model
        │            │            │                │
        └────────────┴────────────┴────────────────┘
                     │
          HuggingFace Hub (model weights)
          HuggingFace Dataset (analytics & feedback)
```

---

## 🤖 Models

### 🎵 Audio Model

Dual-stream architecture; processes raw audio and spectral features **in parallel**.

| Component | Detail |
|---|---|
| **Branch A** | `facebook/wav2vec2-xls-r-300m` (~300M params) |
| **Branch B** | LFCC-LCNN (MFM activation, residual blocks) |
| **Fusion** | Cross-Modal Attention Fusion |
| **LFCC** | 60 coefficients + Δ + ΔΔ → 180 dimensions |
| **Chunking** | 4 sec window, 50% overlap |
| **VAD** | Energy-based, 100 ms minimum speech |

```
Raw audio → VAD → 4s Chunk → [Wav2Vec XLS-R | LFCC-LCNN] → Attention Fusion → P(fake)
```

**Features:** TTA (5 augmentations), MC Dropout (uncertainty), Grad-CAM (interpretability), FGSM/PGD adversarial robustness

---

### 🖼 Image Model

| Component | Detail |
|---|---|
| **Backbone** | `google/siglip2-so400m-patch14-384` |
| **Classifier** | `hidden×2 → 512 → 256 → 64 → 1` |
| **Pooling** | Mean + Max → concat → `hidden×2` |
| **Image Size** | 336×336 px |
| **Augmentation** | 11-technique pool (including FrequencyAugment) |
| **Sampler** | WeightedRandomSampler (class balancing) |
| **Loss** | Focal Loss (γ=1.5, α=0.50) + Label Smoothing |

**Training details:** Progressive unfreeze, SWA, Mixup + CutMix, Cosine Annealing Warm Restarts, Stochastic Depth

---

### 🎬 Video Model

| Component | Detail |
|---|---|
| **Spatial** | `EfficientNet-B5` (RGB) + `FrequencyStreamCNN` (frequency + DCT + optical flow) |
| **Temporal** | 6-layer Transformer Encoder (8 heads) + learnable frame-difference injection |
| **Attention pooling** | MultiheadAttention (mean-pooled query over the sequence) |
| **Frame count** | 12 frames / video |
| **Frequency features** | DoG + Laplacian + RGB inconsistency map (3 ch) + DCT low/mid/high frequency bands (3 ch) → 6 ch total |
| **Optical flow** | Dense (Farneback) — magnitude + angle + consistency, optional (+3 ch) |
| **Fusion** | Two independent learnable gates (RGB and frequency streams each gated separately, then concatenated and projected) |

```
Video → Frame sampling → [EfficientNet-B5 | FreqCNN (DoG+Laplacian+RGB-inconsistency+DCT+OpticalFlow)]
     → Independent Gate Fusion → Temporal Transformer → Attention Pooling (mean query) → P(fake)
```

---

### 📝 Text Model

| Component | Detail |
|---|---|
| **Backbone** | `dbmdz/bert-base-turkish-cased` |
| **Style features** | 12 features (TTR, bigram repetition, AI connectives, etc.) |
| **Architecture** | BERT CLS + Style MLP → Classifier |
| **Threshold** | `AI_THRESHOLD = 0.85` |
| **Minimum text** | 150 words |
| **Language** | Turkish-focused |

**12 style features:** average/std sentence length, type-token ratio, comma/punctuation ratio, paragraph count, bigram repetition, AI connective frequency, word length

---

## 🚀 Installation

### Requirements

```bash
Python >= 3.10
CUDA 11.8+ (recommended)
```

### Environment setup

```bash
git clone https://github.com/YusufCaganCeylan/AI-BUSTER.git
cd AI-BUSTER
pip install -r requirements.txt
```

### Environment variables

```bash
# Define as a .env file or environment variables
export HF_TOKEN="hf_..."        # Model download (read)
export HF_TOKEN_W="hf_..."      # Analytics write (write)
```

### Run the app

```bash
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit
torch
transformers
librosa
opencv-python
albumentations
scikit-learn
huggingface_hub
pdfplumber
python-docx
numpy
```

---

## 📁 File Structure

```
AI-BUSTER/
├── app.py                        # Main Streamlit application
├── logo.jpg                      # Application logo
├── requirements.txt
│
├── training/                     # Training scripts
│   ├── train_audio.py            # Audio model training
│   ├── train_image.py            # Image model training
│   ├── train_video.py            # Video model training
│   └── train_text.py             # Text model training
│
└── README.md
```

---

## 🤗 Hugging Face

| Resource | Link |
|---|---|
| **Demo (Spaces)** | [AI-BUSTER/AI-BUSTER](https://huggingface.co/spaces/AI-BUSTER/AI-BUSTER) |
| **Model weights** | [AI-BUSTER/AI-BUSTER_Models](https://huggingface.co/AI-BUSTER/AI-BUSTER_Models) |
| **Analytics dataset** | `YusufCaganCeylan/AI-BUSTER_Analytics` |

### Files downloaded from HF

| File | Model |
|---|---|
| `audio_model.pth` | Audio detector |
| `video_model.pth` | Video detector |
| `image_model.pth` | Image detector |
| `text_model.pth` | Text detector |
| `scaler.pkl` | Text style feature normalizer |

---

## 🖥 Interface Features

- **Automatic mode detection** — Model selection based on file extension (mp3/wav → Audio, jpg/png → Image, mp4 → Video, txt/pdf → Text)
- **Donut chart** — Visually displays the fake probability
- **Last 5 analysis records** — Session history
- **Feedback module** — Users rate prediction accuracy, saved to HF Dataset
- **Persistent analytics** — `total_analyzed` and `ai_detections` counters stored on HF

---

## 📊 Supported Formats

| Mode | Extensions |
|---|---|
| 🎵 Audio | `mp3`, `wav`, `flac`, `ogg`, `m4a` |
| 🎬 Video | `mp4`, `mov`, `avi`, `mkv`, `webm` |
| 🖼 Image | `jpg`, `jpeg`, `png`, `webp`, `bmp` |
| 📝 Text | `txt`, `pdf`, `docx`, `md` |

---

## ⚠️ Disclaimer

This tool is an assistive detection system. Outputs do not constitute definitive legal or forensic evidence. Expert evaluation is recommended for critical decisions.

---

<div align="center">

**AI-BUSTER** — AI against AI

*Streamlit · PyTorch · HuggingFace · Wav2Vec2 · SigLIP2 · EfficientNet · BERT*

</div>
