# AI-BUSTER 🔍

**AI-BUSTER**, yapay zeka tarafından üretilen içerikleri (deepfake) tespit etmek için geliştirilmiş kapsamlı bir platformdur. Görsel, video ve ses dosyalarını analiz ederek içeriğin AI tarafından üretilip üretilmediğini yüksek doğrulukla belirler.

![AI-BUSTER Logo](logo.jpeg)

## 🌟 Özellikler

### 📸 Görsel Tespit (Image Detection)
- **EFFORT (EFficient rObust deepFake detecTor)** modeli kullanımı
- CLIP-RN50 backbone ile transfer learning
- GenImage veri seti üzerinde eğitilmiş
- Yüksek doğruluk oranı ile AI-generated görsel tespiti

### 🎬 Video Tespit (Video Detection)
- **EfficientNet-B3** tabanlı uzamsal özellik çıkarımı
- **Transformer Encoder** ile zamansal modelleme
- Optical flow analizi ile hareket tutarsızlıklarını tespit
- Frekans analizi (DCT) ile sıkıştırma artefaktlarını belirleme
- Multi-head attention pooling

### 🎵 Ses Tespit (Audio Detection)
- **LFCC (Linear Frequency Cepstral Coefficients)** tabanlı özellik çıkarımı
- Deepfake ses tespiti
- Spoof/Bonafide sınıflandırması
- 16kHz örnekleme hızı ile optimizasyon

## 📋 Gereksinimler

### Sistem Gereksinimleri
- Python 3.8 veya üzeri
- CUDA destekli GPU (önerilen, CPU üzerinde de çalışır)
- 8GB+ RAM (GPU kullanımı için 4GB+ VRAM önerilir)

### Python Kütüphaneleri
Detaylı liste için `requirements.txt` dosyasına bakınız.

## 🚀 Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/YusufCaganCeylan/aidetector.git
cd aidetector
```

### 2. Sanal Ortam Oluşturun (Önerilen)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Model Dosyalarını İndirin
Model Dosyalarını Aşağıdaki Bağlantıdan İndirebilirsini:
https://drive.google.com/drive/folders/1bF6xN905j95U1yEFL5UhfhPaORtY5Lm-?usp=sharing 

## 💻 Kullanım

### Web Arayüzü (Gradio)
```bash
python app.py
```
Tarayıcınızda `http://127.0.0.1:7860` adresini açın.

### Video Modeli Eğitimi
```bash
python Video_train.py
```

#### Veri Seti Yapısı
```
data/
├── real/          # Gerçek videolar
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
└── fake/          # AI-üretimi videolar
    ├── video1.mp4
    ├── video2.avi
    └── ...
```

### Desteklenen Dosya Formatları
- **Görsel:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- **Video:** `.mp4`, `.mov`, `.avi`, `.mkv`
- **Ses:** `.wav` (Yalnızca WAV formatı desteklenir)

## 🏗️ Proje Yapısı

```
aidetector/
├── app.py                      # Ana Gradio web arayüzü
├── Video_train.py              # Video model eğitim scripti
├── requirements.txt            # Python bağımlılıkları
├── README.md                   # Proje dokümantasyonu
├── LICENSE                     # Lisans dosyası
├── logo.jpeg                   # Proje logosu
├── image_detector/             # Görsel tespit modülü
│   ├── detectors/             # Detector mimarileri
│   ├── loss/                  # Loss fonksiyonları
│   ├── metrics/               # Metrik hesaplamaları
│   ├── networks/              # Sinir ağı modelleri
│   ├── training/              # Eğitim konfigürasyonları
│   └── utils/                 # Yardımcı fonksiyonlar
└── test/                      # Test dosyaları
```

## 🎯 Model Mimarileri

### Görsel Tespit Modeli
- **Backbone:** CLIP ResNet-50
- **Framework:** EFFORT (EFficient rObust deepFake detecTor)
- **Input Size:** 224x224
- **Normalization:** CLIP standardı

### Video Tespit Modeli
- **Spatial Extractor:** EfficientNet-B3 (1536 dim)
- **Temporal Encoder:** Transformer (8 heads, 4 layers)
- **Attention Pooling:** Multi-head Attention
- **Classifier:** 4-layer MLP (512→256→64→1)
- **Optical Flow:** Lucas-Kanade method
- **Sequence Length:** 4-10 frame (GPU'ya göre otomatik)

### Ses Tespit Modeli
- **Feature Extraction:** LFCC (40 coefficients)
- **Architecture:** 4-layer MLP
- **Input Dimension:** 12840
- **Sample Rate:** 16kHz
- **Duration:** 4 saniye (64000 sample)

## 📊 Eğitim Özellikleri

### Video Model Eğitimi
- **Otomatik GPU Optimizasyonu:** A100, V100, T4 GPU'ları için özel ayarlar
- **Gradient Checkpointing:** Bellek optimizasyonu
- **Mixed Precision Training:** AMP desteği
- **Data Augmentation:** Albumentations kütüphanesi
- **Loss Functions:** 
  - Label Smoothing BCE
  - Focal Loss
- **Optimizer:** AdamW
- **Scheduler:** Cosine Annealing with Warm Restarts
- **Early Stopping:** Patience-based

### Veri Artırma (Augmentation)
- Horizontal Flip
- Random Brightness/Contrast
- Gaussian Noise
- Random Gamma
- JPEG Compression
- Blur
- Shift/Scale/Rotate

## 📈 Performans

Model performansı, eğitim sırasında otomatik olarak grafiklerle görselleştirilir:
- `training_history.png` - Loss ve accuracy grafikleri
- `confusion_matrix_latest.png` - Confusion matrix
- `detailed_metrics.png` - Detaylı metrikler ve overfitting analizi
- `training.log` - Eğitim logları

## 🔬 Özellik Çıkarımı

### Video İşleme
1. **Frekans Özellikleri:**
   - DCT (Discrete Cosine Transform)
   - Sobel kenar tespiti
   - Renk varyansı analizi

2. **Hareket Özellikleri:**
   - Optical flow (Lucas-Kanade)
   - Hareket büyüklüğü
   - Hareket açısı
   - Tutarlılık skoru

### Görsel İşleme
- CLIP normalizasyonu
- Frekans domeni analizi
- Transfer learning

### Ses İşleme
- LFCC özellik çıkarımı
- 16kHz resampling
- Sabit uzunluk padding

## 🛠️ Yapılandırma

### GPU Otomatik Yapılandırma
Sistem, GPU'nuz otomatik algılar ve en uygun ayarları yapar:

| GPU       | Batch Size | Seq Length | Workers |
|-----------|-----------|------------|---------|
| A100 80GB | 32        | 8          | 4       |
| A100 40GB | 24        | 6          | 4       |
| V100      | 12        | 5          | 2       |
| T4        | 8         | 4          | 2       |
| CPU       | 4         | 3          | 2       |

### Hiperparametre Otomatik Ölçekleme
Veri seti boyutuna göre otomatik ayar:

| Veri Seti   | Epoch | Learning Rate | Label Smoothing |
|-------------|-------|---------------|-----------------|
| <5K         | 5     | 1e-4          | 0.15            |
| 5K-15K      | 30    | 1e-4          | 0.1             |
| 15K-50K     | 50    | 2e-4          | 0.1             |
| 50K+        | 100   | 3e-4          | 0.05            |



## 📝 Lisans

Bu proje [LICENSE](LICENSE) dosyasında belirtilen lisans altında dağıtılmaktadır.


## 📚 Referanslar

- EFFORT: Efficient and Robust deepFake detecTor
- EfficientNet: Rethinking Model Scaling for CNNs
- Attention Is All You Need (Transformer)
- CLIP: Connecting Text and Images

