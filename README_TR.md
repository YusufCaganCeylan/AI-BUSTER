<div align="center">

<img src="logo.jpg" width="90" style="border-radius:16px;" />

# AI-BUSTER

**Çok Modlu Yapay Zeka İçerik Tespit Sistemi**

🇹🇷 Türkçe · 🇬🇧 [English](README.md)

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Demo-AI--BUSTER%20Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/AI-BUSTER/AI-BUSTER)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20Models-AI--BUSTER__Models-FFD21E?style=for-the-badge)](https://huggingface.co/AI-BUSTER/AI-BUSTER_Models)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

*Ses · Görsel · Video · Metin — tek arayüzde dört modda gerçek zamanlı deepfake tespiti*

</div>

---

## ✨ Neler Değişti?

> Bu sürüm, projenin **Gradio → Streamlit** geçişini, dört bağımsız sinir ağı modelini ve Hugging Face Spaces üzerinde çalışan canlı demoyu kapsamaktadır.

| Özellik | Eski (Gradio) | Yeni (Streamlit) |
|---|---|---|
| Arayüz | Gradio Blocks | Streamlit + özel CSS teması |
| Model yükleme | Yerel dosya | HuggingFace Hub (`hf_hub_download`) |
| Modeller | Ses + Görsel + Video | **Ses + Görsel + Video + Metin** |
| Analitik | Yok | HF Dataset üzerinde kalıcı kayıt |
| Demo | — | [🤗 HF Spaces](https://huggingface.co/spaces/AI-BUSTER/AI-BUSTER) |

---

## 🏗 Sistem Mimarisi

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Arayüzü                  │
│         (Dosya yükle  ·  Metin gir)                 │
└────────────────────┬────────────────────────────────┘
                     │  uzantıya göre otomatik yönlendirme
        ┌────────────┼────────────┬────────────────┐
        ▼            ▼            ▼                ▼
   🎵 Ses        🖼 Görsel    🎬 Video         📝 Metin
     Modeli          Modeli      Modeli            Modeli
        │            │            │                │
        └────────────┴────────────┴────────────────┘
                     │
          HuggingFace Hub (model ağırlıkları)
          HuggingFace Dataset (analitik & feedback)
```

---

## 🤖 Modeller

### 🎵 Ses Modeli

Çift akışlı mimari; ham ses ve spektral özellikleri **paralel** işler.

| Bileşen | Detay |
|---|---|
| **Kol A** | `facebook/wav2vec2-xls-r-300m` (~300M param) |
| **Kol B** | LFCC-LCNN (MFM aktivasyon, residual bloklar) |
| **Füzyon** | Cross-Modal Attention Fusion |
| **LFCC** | 60 katsayı + Δ + ΔΔ → 180 boyut |
| **Chunking** | 4 sn pencere, %50 örtüşme |
| **VAD** | Enerji tabanlı, 100 ms minimum konuşma |

```
Ham ses → VAD → 4s Chunk → [Wav2Vec XLS-R | LFCC-LCNN] → Attention Fusion → P(fake)
```

**Özellikler:** TTA (5 augmentation), MC Dropout (uncertainty), Grad-CAM (yorumlanabilirlik), FGSM/PGD adversarial robustness

---

### 🖼 Görsel Modeli

| Bileşen | Detay |
|---|---|
| **Backbone** | `google/siglip2-so400m-patch14-384` |
| **Classifier** | `hidden×2 → 512 → 256 → 64 → 1` |
| **Pooling** | Mean + Max → concat → `hidden×2` |
| **Image Size** | 336×336 px |
| **Augmentation** | 11-teknik havuz (FrequencyAugment dahil) |
| **Sampler** | WeightedRandomSampler (sınıf dengesi) |
| **Loss** | Focal Loss (γ=1.5, α=0.50) + Label Smoothing |

**Eğitim detayları:** Progressive unfreeze, SWA, Mixup + CutMix, Cosine Annealing Warm Restarts, Stochastic Depth, 

---

### 🎬 Video Modeli

| Bileşen | Detay |
|---|---|
| **Spatial** | `EfficientNet-B5` (RGB) + `FrequencyStreamCNN` (frekans) |
| **Temporal** | 6 katmanlı Transformer Encoder (8 kafa) |
| **Dikkat havuzu** | MultiheadAttention (öğrenilebilir sorgu) |
| **Çerçeve sayısı** | 12 çerçeve / video |
| **Frekans özellikleri** | DoG + Laplacian + RGB tutarsızlık haritası |
| **Füzyon** | Öğrenilebilir gate (RGB ⊙ gate + freq ⊙ (1-gate)) |

```
Video → Çerçeve örnekleme → [EfficientNet-B5 | FreqCNN] → Gate Fusion
     → Temporal Transformer → Attention Pooling → P(fake)
```

---

### 📝 Metin Modeli

| Bileşen | Detay |
|---|---|
| **Backbone** | `dbmdz/bert-base-turkish-cased` |
| **Stil özellikleri** | 12 adet (TTR, bigram tekrar, AI bağlaçları vb.) |
| **Mimari** | BERT CLS + Stil MLP → Classifier |
| **Eşik** | `AI_THRESHOLD = 0.85` |
| **Minimum metin** | 150 kelime |
| **Dil** | Türkçe odaklı |

**12 stil özelliği:** ortalama/std cümle uzunluğu, type-token ratio, virgül/noktalama oranı, paragraf sayısı, bigram tekrar, AI bağlaç sıklığı, kelime uzunluğu

---

## 🚀 Kurulum

### Gereksinimler

```bash
Python >= 3.10
CUDA 11.8+ (önerilir)
```

### Ortam kurulumu

```bash
git clone https://github.com/YusufCaganCeylan/AI-BUSTER.git
cd AI-BUSTER
pip install -r requirements.txt
```

### Ortam değişkenleri

```bash
# .env dosyası veya ortam değişkeni olarak tanımlayın
export HF_TOKEN="hf_..."        # Model indirme (okuma)
export HF_TOKEN_W="hf_..."      # Analitik yazma (write)
```

### Uygulamayı başlatın

```bash
streamlit run app.py
```

---

## 📦 Bağımlılıklar

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

## 📁 Dosya Yapısı

```
AI-BUSTER/
├── app.py                        # Ana Streamlit uygulaması
├── logo.jpg                      # Uygulama logosu
├── requirements.txt
│
├── training/                     # Eğitim scriptleri
│   ├── train_audio.py            # Ses modeli eğitimi
│   ├── train_image.py            # Görsel modeli eğitimi
│   ├── train_video.py            # Video modeli eğitimi
│   └── train_text.py             # Metin modeli eğitimi
│
└── README.md
```

---

## 🤗 Hugging Face

| Kaynak | Bağlantı |
|---|---|
| **Demo (Spaces)** | [AI-BUSTER/AI-BUSTER](https://huggingface.co/spaces/AI-BUSTER/AI-BUSTER) |
| **Model ağırlıkları** | [AI-BUSTER/AI-BUSTER_Models](https://huggingface.co/AI-BUSTER/AI-BUSTER_Models) |
| **Analitik dataset** | `YusufCaganCeylan/AI-BUSTER_Analytics` |

### HF'den indirilen dosyalar

| Dosya | Model |
|---|---|
| `audio_model.pth` | Ses dedektörü |
| `video_model.pth` | Video dedektörü |
| `image_model.pth` | Görsel dedektörü |
| `text_model.pth` | Metin dedektörü |
| `scaler.pkl` | Metin stil özelliği normalize edici |

---

## 🖥 Arayüz Özellikleri

- **Otomatik mod tespiti** — Dosya uzantısına göre model seçimi (mp3/wav → Ses, jpg/png → Görsel, mp4 → Video, txt/pdf → Metin)
- **Donut grafik** — Sahte olasılığını görsel olarak gösterir
- **Son 5 analiz kaydı** — Oturum geçmişi
- **Geri bildirim modülü** — Tahmin doğruluğunu kullanıcı değerlendirir, HF Dataset'e kaydedilir
- **Kalıcı analitik** — `total_analyzed` ve `deepfake_hits` sayaçları HF üzerinde saklanır

---

## 📊 Desteklenen Formatlar

| Mod | Uzantılar |
|---|---|
| 🎵 Ses | `mp3`, `wav`, `flac`, `ogg`, `m4a` |
| 🎬 Video | `mp4`, `mov`, `avi`, `mkv`, `webm` |
| 🖼 Görsel | `jpg`, `jpeg`, `png`, `webp`, `bmp` |
| 📝 Metin | `txt`, `pdf`, `docx`, `md` |

---

## ⚠️ Sorumluluk Reddi

Bu araç yardımcı bir tespit sistemidir. Çıktılar kesin hukuki ya da adli kanıt niteliği taşımaz. Kritik kararlar için uzman değerlendirmesi önerilir.

---

<div align="center">

**AI-BUSTER** — Yapay zekaya karşı yapay zeka

*Streamlit · PyTorch · HuggingFace · Wav2Vec2 · SigLIP2 · EfficientNet · BERT*

</div>
