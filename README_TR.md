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

*Ses · Görüntü · Video · Metin — dört modda, tek arayüzde gerçek zamanlı yapay zeka tespiti*

</div>

---

## ✨ Yenilikler

> Bu sürüm, projenin **Gradio → Streamlit** geçişini, dört bağımsız sinir ağı modelini ve Hugging Face Spaces üzerinde çalışan canlı bir demoyu kapsıyor.

| Özellik | Eski (Gradio) | Yeni (Streamlit) |
|---|---|---|
| Arayüz | Gradio Blocks | Streamlit + özel CSS teması |
| Model yükleme | Yerel dosya | HuggingFace Hub (`hf_hub_download`) |
| Modeller | Ses + Görüntü + Video | **Ses + Görüntü + Video + Metin** |
| Analitik | Yok | HF Dataset üzerinde kalıcı loglama |
| Demo | — | [🤗 HF Spaces](https://huggingface.co/spaces/AI-BUSTER/AI-BUSTER) |

---

## 🏗 Sistem Mimarisi

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Arayüzü                  │
│         (Dosya Yükle  ·  Metin Gir)                 │
└────────────────────┬────────────────────────────────┘
                     │  uzantıya göre otomatik yönlendirme
        ┌────────────┼────────────┬────────────────┐
        ▼            ▼            ▼                ▼
   🎵 Ses         🖼 Görüntü   🎬 Video         📝 Metin
     Modeli         Modeli      Modeli           Modeli
        │            │            │                │
        └────────────┴────────────┴────────────────┘
                     │
          HuggingFace Hub (model ağırlıkları)
          HuggingFace Dataset (analitik & geri bildirim)
```

---

## 🤖 Modeller

### 🎵 Ses Modeli

Çift akışlı mimari; ham ses ve spektral özellikleri **paralel** olarak işler.

| Bileşen | Detay |
|---|---|
| **Kol A** | `facebook/wav2vec2-xls-r-300m` (~300M parametre) |
| **Kol B** | LFCC-LCNN (MFM aktivasyonu, residual bloklar) |
| **Füzyon** | Cross-Modal Attention Fusion |
| **LFCC** | 60 katsayı + Δ + ΔΔ → 180 boyut |
| **Parçalama** | 4 saniyelik pencere, %50 örtüşme |
| **VAD** | Enerji tabanlı, minimum 100 ms konuşma |

```
Ham ses → VAD → 4s Parça → [Wav2Vec XLS-R | LFCC-LCNN] → Attention Fusion → P(sahte)
```

**Özellikler:** TTA (5 augmentasyon), MC Dropout (belirsizlik), Grad-CAM (yorumlanabilirlik), FGSM/PGD adversarial dayanıklılık

---

### 🖼 Görüntü Modeli

| Bileşen | Detay |
|---|---|
| **Omurga** | `google/siglip2-so400m-patch14-384` |
| **Sınıflandırıcı** | `hidden×2 → 512 → 256 → 64 → 1` |
| **Pooling** | Mean + Max → concat → `hidden×2` |
| **Görüntü Boyutu** | 336×336 px |
| **Augmentasyon** | 11 teknikten oluşan havuz (FrequencyAugment dahil) |
| **Örnekleyici** | WeightedRandomSampler (sınıf dengeleme) |
| **Kayıp Fonksiyonu** | Focal Loss (γ=1.5, α=0.50) + Label Smoothing |

**Eğitim detayları:** Aşamalı çözme (progressive unfreeze), SWA, Mixup + CutMix, Cosine Annealing Warm Restarts, Stochastic Depth

---

### 🎬 Video Modeli

| Bileşen | Detay |
|---|---|
| **Uzamsal (Spatial)** | `EfficientNet-B5` (RGB) + `FrequencyStreamCNN` (frekans + DCT + optical flow) |
| **Zamansal (Temporal)** | 6 katmanlı Transformer Encoder (8 head) + öğrenilebilir kare-farkı (frame-difference) enjeksiyonu |
| **Attention pooling** | MultiheadAttention (sekans üzerinden ortalaması alınmış query) |
| **Kare sayısı** | 12 kare / video |
| **Frekans özellikleri** | DoG + Laplacian + RGB tutarsızlık haritası (3 kanal) + DCT düşük/orta/yüksek frekans bantları (3 kanal) → toplam 6 kanal |
| **Optical flow** | Dense (Farneback) — büyüklük + açı + tutarlılık, opsiyonel (+3 kanal) |
| **Füzyon** | İki bağımsız öğrenilebilir gate (RGB ve frekans akışları ayrı ayrı gate'lenir, ardından birleştirilip projekte edilir) |

```
Video → Kare örnekleme → [EfficientNet-B5 | FreqCNN (DoG+Laplacian+RGB-tutarsızlık+DCT+OpticalFlow)]
     → Bağımsız Gate Füzyonu → Zamansal Transformer → Attention Pooling (ortalama query) → P(sahte)
```

---

### 📝 Metin Modeli

| Bileşen | Detay |
|---|---|
| **Omurga** | `dbmdz/bert-base-turkish-cased` |
| **Stil özellikleri** | 12 özellik (TTR, bigram tekrarı, AI bağlaçları vb.) |
| **Mimari** | BERT CLS + Stil MLP → Sınıflandırıcı |
| **Eşik** | `AI_THRESHOLD = 0.85` |
| **Minimum metin** | 150 kelime |
| **Dil** | Türkçe odaklı |

**12 stil özelliği:** ortalama/standart sapma cümle uzunluğu, type-token oranı, virgül/noktalama oranı, paragraf sayısı, bigram tekrarı, AI bağlaç sıklığı, kelime uzunluğu

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
export HF_TOKEN="hf_..."        # Model indirme (read)
export HF_TOKEN_W="hf_..."      # Analitik yazma (write)
```

### Uygulamayı çalıştırma

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
│   ├── train_image.py            # Görüntü modeli eğitimi
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
| **Analitik veri seti** | `YusufCaganCeylan/AI-BUSTER_Analytics` |

### HF'den indirilen dosyalar

| Dosya | Model |
|---|---|
| `audio_model.pth` | Ses dedektörü |
| `video_model.pth` | Video dedektörü |
| `image_model.pth` | Görüntü dedektörü |
| `text_model.pth` | Metin dedektörü |
| `scaler.pkl` | Metin stil özelliği normalizeri |

---

## 🖥 Arayüz Özellikleri

- **Otomatik mod tespiti** — Dosya uzantısına göre model seçimi (mp3/wav → Ses, jpg/png → Görüntü, mp4 → Video, txt/pdf → Metin)
- **Donut grafik** — Sahtelik olasılığını görsel olarak gösterir
- **Son 5 analiz kaydı** — Oturum geçmişi
- **Geri bildirim modülü** — Kullanıcılar tahmin doğruluğunu puanlar, HF Dataset'e kaydedilir
- **Kalıcı analitik** — `total_analyzed` ve `ai_detections` sayaçları HF üzerinde saklanır

---

## 📊 Desteklenen Formatlar

| Mod | Uzantılar |
|---|---|
| 🎵 Ses | `mp3`, `wav`, `flac`, `ogg`, `m4a` |
| 🎬 Video | `mp4`, `mov`, `avi`, `mkv`, `webm` |
| 🖼 Görüntü | `jpg`, `jpeg`, `png`, `webp`, `bmp` |
| 📝 Metin | `txt`, `pdf`, `docx`, `md` |

---

## ⚠️ Sorumluluk Reddi

Bu araç, yardımcı bir tespit sistemidir. Çıktılar kesin bir hukuki veya adli delil niteliği taşımaz. Kritik kararlar için uzman değerlendirmesi önerilir.

---

<div align="center">

**AI-BUSTER** — Yapay zekaya karşı yapay zeka

*Streamlit · PyTorch · HuggingFace · Wav2Vec2 · SigLIP2 · EfficientNet · BERT*

</div>
