# -*- coding: utf-8 -*-
"""
SigLIP2 Görsel Deepfake Dedektörü — v9 (F1 > 0.90 Hedefli)
============================================================
Bu script, Google'ın SigLIP2 modelini temel alarak gerçek/sahte görsel
sınıflandırması yapar. v8'e kıyasla kritik iyileştirmeler içerir:

  ✅ [KRİTİK 1] unfreeze_last_n: 4 → 8  (daha fazla backbone açık)
  ✅ [KRİTİK 2] Backbone LR çarpanı: 0.10 → 0.25  (2e-5 → 5e-5)
  ✅ [KRİTİK 3] early_stopping_patience: 12 → 20
  ✅ [EKSTRA 1] swa_start_epoch: 15 → 8  (SWA artık devreye giriyor)
  ✅ [EKSTRA 2] image_size: 224 → 336, batch_size: 32 → 16
  ✅ [EKSTRA 3] GeM pooling eklendi (mean+gem+max → hidden*3)
  ✅ [EKSTRA 4] Classifier güncellendi: hidden*3 → 768 → 384 → 128 → 1
  ✅ [EKSTRA 5] cosine_T0: 7 → 10 (daha uzun restart periyodu)
  ✅ [EKSTRA 6] min_auc_to_save: 0.72 → 0.85
"""

import os, random, warnings, logging, shutil, time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")  # GPU ortamında GUI olmadan grafik kaydetmek için
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

# PyTorch GPU optimizasyonları — eğitim hızını artırır
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

EPSILON = 1e-7  # Sıfıra bölünmeyi önlemek için küçük sayı

# ============================================================
# ⚙️  KONFİGÜRASYON
# ============================================================
# Google Drive ve yerel SSD yolları — Colab ortamı için optimize edilmiş
DRIVE_DATASET = "/content/drive/MyDrive/imagedatasetv2"
LOCAL_DATASET = "/content/local_dataset"  # SSD'ye kopyalanacak yol (daha hızlı I/O)
PROJECT_DIR   = "/content/drive/MyDrive/siglip2_projectpopo3"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "epoch_plots"), exist_ok=True)

CONFIG = {
    # ── Model ──────────────────────────────────────────────
    "model_name":        "google/siglip2-so400m-patch14-384",  # ~400M param backbone
    "epochs":            60,
    "batch_size":        16,            # 32 → 16: image_size 336 için VRAM tüketimi azaltıldı
    "grad_accum_steps":  2,             # Efektif batch = 16×2 = 32 korunuyor
    "use_amp":           True,          # Otomatik karışık hassasiyet (FP16) — hız + bellek kazanımı
    "use_grad_ckpt":     True,          # Gradient checkpointing — VRAM tasarrufu
    "learning_rate":     2e-4,          # Classifier baş LR'si
    "weight_decay":      0.08,
    "image_size":        336,           # 224 → 336: model 384 için tasarlandı, 336 daha iyi uzlaşım

    # ── Dataset ────────────────────────────────────────────
    "train_real_dir": os.path.join(LOCAL_DATASET, "train", "real"),
    "train_fake_dir": os.path.join(LOCAL_DATASET, "train", "fake"),
    "val_real_dir":   os.path.join(LOCAL_DATASET, "val",   "real"),
    "val_fake_dir":   os.path.join(LOCAL_DATASET, "val",   "fake"),
    "test_real_dir":  os.path.join(LOCAL_DATASET, "test",  "real"),
    "test_fake_dir":  os.path.join(LOCAL_DATASET, "test",  "fake"),
    "test_split_ratio":  0.15,  # Otomatik split modunda %15 test ayrılır
    "val_split_ratio":   0.15,  # Otomatik split modunda %15 val ayrılır

    # ── Kayıt ──────────────────────────────────────────────
    "save_path":            os.path.join(PROJECT_DIR, "best_model_v9.pth"),
    "local_epoch_plot_dir": os.path.join(PROJECT_DIR, "epoch_plots"),
    "project_dir":          PROJECT_DIR,

    # ── DataLoader ─────────────────────────────────────────
    "num_workers":  4,          # Paralel veri yükleme işçisi sayısı
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "cache_images": False,      # True yapılırsa tüm dataset RAM'e yüklenir (hızlanır ama bellek ister)

    # ── Backbone Dondurma / Açma Stratejisi ────────────────
    "freeze_backbone":    True,
    "unfreeze_last_n":    8,            # v8'de 4'tü; 8 yaparak backbone'un yarısını açtık
    "progressive_unfreeze": True,       # Eğitim boyunca kademeli olarak daha fazla blok açılır
    "progressive_unfreeze_schedule": {
        0:  8,    # Baştan: son 8 blok açık
        3:  12,   # 3. epoch'tan itibaren: son 12 blok açık
        8:  16,   # 8. epoch'tan itibaren: son 16 blok açık
        15: 20,   # 15. epoch'tan itibaren: son 20 blok açık
    },

    # ── Augmentation ───────────────────────────────────────
    "use_distortion_aug": True,  # Bozulma augmentasyonları (blur, noise, compression vb.)
    "use_mixup":          True,  # İki görseli harmanlama — genelleştirme artırır
    "mixup_alpha":        0.3,   # Beta dağılımı parametresi (0.3 → ılımlı harmanlama)
    "cutmix_prob":        0.2,   # Mixup yerine CutMix uygulama olasılığı

    # ── Regularizasyon ─────────────────────────────────────
    "early_stopping_patience":  20,   # v8'de 12'ydi; cosine restart'ların tamamını görmek için 20
    "early_stopping_min_delta": 5e-4,
    "label_smoothing":          0.05, # Aşırı güven (overconfidence) önler
    "dropout_head":             0.30,
    "dropout_mid":              0.15,
    "warmup_epochs":            2,    # İlk 2 epoch'ta LR lineer ısınma
    "stochastic_depth_prob":    0.15, # Stokastik derinlik — random layer skip

    # ── Kayıp Fonksiyonu ───────────────────────────────────
    "use_focal_loss": True,    # Focal loss — zor örneklere daha fazla ağırlık verir
    "focal_gamma":    1.5,     # Kolayca sınıflanan örneklerin ağırlığını azaltma faktörü
    "focal_alpha":    0.50,    # Sınıf dengesi için alfa (0.5 = nötr)

    # ── Test-Time Augmentation ─────────────────────────────
    "use_tta":   True,  # Test sırasında 5 farklı augmentation ile tahmin ortalaması alınır
    "tta_n_aug": 5,

    # ── R-Drop ─────────────────────────────────────────────
    "use_rdrop":   False,  # Aynı girdiyi iki kez forward edip KL divergence ekler (kapalı)
    "rdrop_alpha": 0.5,

    # ── Füzyon ─────────────────────────────────────────────
    "fusion_type":        "simple",
    "feature_dropout_p":  0.20,   # Özellik vektörlerine uygulanan dropout

    # ── Öğrenme Oranı Zamanlayıcısı ────────────────────────
    "use_cosine_restarts": True,
    "cosine_T0":           10,    # İlk restart periyodu (epoch) — v8'de 7'ydi
    "cosine_T_mult":       2,     # Her restart sonrası periyot iki katına çıkar

    # ── Gürültü ────────────────────────────────────────────
    "noise_std": 0.02,  # Özellik vektörlerine eklenen Gaussian gürültü (regularization)

    # ── Stochastic Weight Averaging ────────────────────────
    "use_swa":         True,
    "swa_start_epoch": 8,    # v8'de 15'ti; erken başlatarak daha fazla epoch SWA kapsar
    "swa_lr":          1e-6, # SWA için sabit düşük LR

    # ── Dengeli Örnekleyici ────────────────────────────────
    "use_balanced_sampler": True,  # Sınıf dengesizliğini WeightedRandomSampler ile giderir

    # ── Model Kayıt Eşiği ──────────────────────────────────
    "min_auc_to_save": 0.85,  # Bu AUC değerinin altındaki modeller kaydedilmez (v8: 0.72)

    # ── Torch Compile (deneysel) ───────────────────────────
    "use_compile": False,  # True yapılırsa PyTorch 2.0 graph compilation aktif olur

    # ── EMA Threshold (kapalı) ─────────────────────────────
    "ema_threshold_alpha": 0.0,  # 0 = EMA eşik smoothing yok; sadece dinamik threshold kullanılır
}


# ============================================================
# 🚀 DRIVE → SSD KOPYALAMA
# ============================================================
def copy_dataset_to_ssd(drive_root: str, local_root: str, force: bool = False):
    """
    Dataset'i Google Drive'dan yerel SSD'ye kopyalar.
    Drive I/O çok yavaş olduğu için bu adım eğitim hızını önemli ölçüde artırır.

    Args:
        drive_root: Google Drive'daki kaynak klasör yolu
        local_root: Hedef SSD yolu (/content/ altı)
        force: True ise mevcut SSD kopyasını silerek yeniden kopyalar
    """
    if os.path.exists(local_root) and not force:
        n = sum(len(files) for _, _, files in os.walk(local_root))
        if n > 0:
            print(f"✅ Dataset zaten SSD'de: {local_root} ({n} dosya)")
            return

    if not os.path.exists(drive_root):
        print(f"⚠️  Drive klasörü bulunamadı: {drive_root}")
        return

    print(f"📂 Dataset Drive → SSD kopyalanıyor...")
    t0 = time.time()
    if os.path.exists(local_root):
        shutil.rmtree(local_root)
    shutil.copytree(drive_root, local_root)
    n = sum(len(files) for _, _, files in os.walk(local_root))
    print(f"✅ Kopyalama tamamlandı: {n} dosya, {time.time()-t0:.1f} saniye\n")


# ============================================================
# 🔥 AUGMENTATION
# ============================================================
class FrequencyAugment(A.ImageOnlyTransform):
    """
    Frekans tabanlı özel augmentation.
    Görsel sinyaline düşey çizgiler (grid pattern) ekleyerek
    modelin frekans artefaktlarına karşı daha dayanıklı olmasını sağlar.
    Deepfake görsellerin üretim sırasında bıraktığı frekans izlerine benzer yapılar oluşturur.
    """
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        if random.random() < 0.5:
            # 8 piksel aralıklı yatay çizgiler ekle
            grid = np.zeros_like(img, dtype=np.float32)
            for i in range(0, h, 8):
                grid[i:min(i+1, h), :] += random.uniform(1, 5)
            img = np.clip(img.astype(np.float32) + grid * random.uniform(0.1, 0.4), 0, 255).astype(np.uint8)
        return img

    def get_transform_init_args_names(self):
        return ()


# 11 farklı augmentation tekniğinin havuzu
# Eğitimde bu havuzdan rastgele seçim yapılır
FAST_DISTORTION_POOL = [
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),                                        # Bulanıklaştırma
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),                                      # Gaussian gürültü
    A.ImageCompression(quality_lower=40, quality_upper=85, p=1.0),                    # JPEG sıkıştırma artefaktı
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),     # Parlaklık/kontrast
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),  # Renk dönüşümü
    A.RandomGamma(gamma_limit=(70, 130), p=1.0),                                      # Gamma düzeltme
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),                       # Keskinleştirme
    A.CoarseDropout(max_holes=6, max_height=32, max_width=32, min_holes=1, p=1.0),  # Rastgele bölge silme
    A.HorizontalFlip(p=1.0),                                                          # Yatay ayna
    A.Rotate(limit=10, p=1.0),                                                        # Küçük açıda döndürme
    FrequencyAugment(p=1.0),                                                          # Özel frekans aug.
]

# ImageNet normalize istatistikleri — pretrained model için standart
_NORM = [
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
]

def get_train_transform(image_size: int = 336):
    """
    Eğitim için dinamik augmentation pipeline oluşturur.
    3 zorluk seviyesi arasından ağırlıklı rastgele seçim yapar:
      - Level 1 (ağırlık %40): Hiç augmentation yok — temiz görsel
      - Level 2 (ağırlık %45): 1-3 teknik — orta düzey bozulma
      - Level 3 (ağırlık %15): 3-5 teknik — ağır bozulma
    """
    resize = [A.Resize(image_size, image_size)]
    level  = random.choices([1, 2, 3], weights=[0.40, 0.45, 0.15])[0]
    if level == 1:
        aug = []
    elif level == 2:
        aug = random.sample(FAST_DISTORTION_POOL, random.randint(1, 3))
    else:
        aug = random.sample(FAST_DISTORTION_POOL, random.randint(3, 5))
    return A.Compose(resize + aug + _NORM)


# Validation/test için sabit, augmentation içermeyen transform
VAL_TRANSFORM = A.Compose([
    A.Resize(336, 336),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def get_tta_transforms() -> List[A.Compose]:
    """
    Test-Time Augmentation için 5 farklı transform döndürür.
    Her transform, aynı görselin farklı bir versiyonunu üretir.
    5 tahminin ortalaması alınarak daha kararlı sonuç elde edilir.
    """
    norm = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
    return [
        A.Compose([A.Resize(336, 336)] + norm),                                              # Orijinal
        A.Compose([A.Resize(336, 336), A.HorizontalFlip(p=1.0)] + norm),                   # Ayna
        A.Compose([A.Resize(336, 336), A.RandomBrightnessContrast(0.1, 0.1, p=1.0)] + norm), # Parlaklık
        A.Compose([A.Resize(336, 336), A.GaussianBlur(blur_limit=(3, 3), p=1.0)] + norm),  # Bulanık
        A.Compose([A.Resize(336, 336), A.Rotate(limit=5, p=1.0)] + norm),                  # Hafif döndürme
    ]


# ============================================================
# 📁 DATASET
# ============================================================
class AIDetectionDataset(Dataset):
    """
    Gerçek/Sahte görsel sınıflandırması için PyTorch Dataset.

    Dizin yapısı beklentisi:
        real_dir/  → gerçek görsel dosyaları (label=1)
        fake_dir/  → sahte/deepfake görsel dosyaları (label=0)

    Desteklenen formatlar: .jpg, .jpeg, .png, .webp, .bmp

    Not: cache_images=True ile tüm görseller RAM'e yüklenir.
    Büyük datasetlerde bellek taşmasına dikkat edin.
    """
    VALID_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    def __init__(self, real_dir, fake_dir, split="train", config=None, samples=None):
        self.config   = config
        self.split    = split
        self.is_train = (split == "train")
        self.cache    = {}  # Opsiyonel RAM cache (path → numpy array)

        if samples is not None:
            # Dışarıdan hazır örnek listesi verilmişse kullan (source-aware split için)
            self.samples = samples
        else:
            # Dizinlerden otomatik örnek listesi oluştur
            self.samples = []
            for directory, label in [(real_dir, 1), (fake_dir, 0)]:
                if directory and os.path.exists(directory):
                    for f in sorted(os.listdir(directory)):
                        if f.lower().endswith(self.VALID_EXT):
                            self.samples.append((os.path.join(directory, f), label))
                else:
                    logger.warning(f"⚠️  Klasör bulunamadı: {directory}")
            if self.is_train:
                random.shuffle(self.samples)  # Eğitim setini karıştır

        # Sınıf dağılımını logla
        real_c = sum(1 for _, l in self.samples if l == 1)
        fake_c = sum(1 for _, l in self.samples if l == 0)
        print(f"{split.upper():5}: {len(self.samples):>5} görsel  |  Real:{real_c}  Fake:{fake_c}")

        # İstenirse tüm görseller RAM'e yükle (disk I/O darboğazını ortadan kaldırır)
        if config and config.get("cache_images", False):
            print(f"   💾 RAM'e yükleniyor ({split})...")
            for path, _ in tqdm(self.samples, leave=False):
                try:
                    self.cache[path] = np.array(Image.open(path).convert("RGB"))
                except Exception:
                    sz = config.get("image_size", 336)
                    self.cache[path] = np.zeros((sz, sz, 3), dtype=np.uint8)
            print(f"   ✅ {len(self.cache)} görsel RAM'de")

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        """WeightedRandomSampler için tüm etiketleri döndürür."""
        return [label for _, label in self.samples]

    def _load(self, path):
        """Görsel yükler — önce cache'e bakar, yoksa diskten okur."""
        if path in self.cache:
            return self.cache[path]
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            # Bozuk görsel durumunda siyah piksel döndür
            sz = self.config.get("image_size", 336)
            return np.zeros((sz, sz, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self._load(path)
        # Eğitim setinde dinamik augmentation, val/test'te sabit transform
        if self.is_train:
            t = get_train_transform(self.config.get("image_size", 336))
        else:
            t = VAL_TRANSFORM
        return t(image=image)["image"], torch.tensor(label, dtype=torch.float32)


# ============================================================
# ⚖️  BALANCED SAMPLER
# ============================================================
def make_balanced_sampler(dataset):
    """
    Sınıf dengesizliğini telafi eden WeightedRandomSampler oluşturur.
    Az sayıda örneği olan sınıf (genellikle fake) daha sık örneklenir.
    Bu, modelin baskın sınıfa aşırı uymadan her iki sınıfı eşit öğrenmesini sağlar.
    """
    labels  = dataset.get_labels()
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]  # Sınıf sıklığının tersi = ağırlık
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(dataset),
        replacement=True,  # Yeniden örneklemeye izin ver
    )


# ============================================================
# 🔀 SOURCE-AWARE SPLIT
# ============================================================
def source_aware_split(all_samples, val_ratio=0.15, test_ratio=0.15):
    """
    Aynı kaynak klasörden (örn. aynı video karesinden alınan görseller)
    örneklerin farklı setlere sızmasını önler.

    Klasik dosya bazlı split'te aynı videonun farklı kareleri train ve test'e
    düşebilir — bu "veri sızıntısı" (data leakage) oluşturur ve metrikleri şişirir.
    Bu fonksiyon klasör bazlı gruplama yaparak bu sorunu çözer.
    """
    # Dosyaları kaynak klasörlerine göre grupla
    groups = {}
    for path, label in all_samples:
        gid = os.path.dirname(path)
        groups.setdefault(gid, []).append((path, label))

    keys = list(groups.keys())
    random.shuffle(keys)
    n  = len(keys)
    nt = max(1, int(n * test_ratio))
    nv = max(1, int(n * val_ratio))
    nt2 = n - nt - nv

    # Grup bazında train/val/test ayrımı
    train = [s for k in keys[:nt2]        for s in groups[k]]
    val   = [s for k in keys[nt2:nt2+nv]  for s in groups[k]]
    test  = [s for k in keys[nt2+nv:]     for s in groups[k]]
    return train, val, test


def build_datasets(config):
    """
    İki modda çalışır:
      - Mod A: Hazır train/val/test klasörleri varsa doğrudan kullanır
      - Mod B: Tüm veriyi alıp source-aware split ile otomatik böler
    """
    tr = config.get("test_real_dir", "")
    tf = config.get("test_fake_dir", "")
    has_test = os.path.exists(tr) and os.path.exists(tf)

    if has_test:
        logger.info("📂 Mod A: hazır train/val/test klasörleri")
        return (
            AIDetectionDataset(config["train_real_dir"], config["train_fake_dir"], "train", config),
            AIDetectionDataset(config["val_real_dir"],   config["val_fake_dir"],   "val",   config),
            AIDetectionDataset(tr, tf, "test", config),
        )

    logger.info("📂 Mod B: otomatik split")
    all_s = []
    for d, lbl in [(config["train_real_dir"], 1), (config["train_fake_dir"], 0),
                   (config["val_real_dir"],   1), (config["val_fake_dir"],   0)]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(AIDetectionDataset.VALID_EXT):
                    all_s.append((os.path.join(d, f), lbl))

    tr_s, va_s, te_s = source_aware_split(
        all_s, config.get("val_split_ratio", 0.15), config.get("test_split_ratio", 0.15))
    return (
        AIDetectionDataset(None, None, "train", config, tr_s),
        AIDetectionDataset(None, None, "val",   config, va_s),
        AIDetectionDataset(None, None, "test",  config, te_s),
    )


# ============================================================
# ⏹️  EARLY STOPPING
# ============================================================
class EarlyStopping:
    """
    AUC ve F1 metriklerinin ağırlıklı ortalamasına göre erken durdurma uygular.
    Skor = 1 - (0.6×AUC + 0.4×F1): küçük = iyi.
    'patience' epoch boyunca iyileşme olmazsa eğitimi durdurur.
    """
    def __init__(self, patience=20, min_delta=5e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")
        self.stop      = False

    def __call__(self, auc, f1):
        metric = 1.0 - (0.6 * auc + 0.4 * f1)  # Düşük = daha iyi
        if metric < self.best - self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ============================================================
# 🔓 BACKBONE UNFREEZE
# ============================================================
def unfreeze_last_n_blocks(vision_model, n):
    """
    SigLIP2 vision encoder'ının son n bloğunu eğitime açar,
    geri kalanları dondurulmuş bırakır.

    Progressive unfreeze stratejisi: eğitim ilerledikçe
    daha fazla blok açılır (CONFIG'deki schedule'a göre).
    Bu yaklaşım:
      - Başta: Yalnızca classifier kafası eğitilir (hızlı öğrenme)
      - Sonra: Backbone ince ayarı ile daha derin uyum sağlanır
    """
    if n <= 0:
        return
    try:
        # Önce tüm parametreleri dondur
        for p in vision_model.parameters():
            p.requires_grad = False
        # Son n bloğu aç
        blocks = list(vision_model.encoder.layers)
        for block in blocks[max(0, len(blocks) - n):]:
            for p in block.parameters():
                p.requires_grad = True
        # Post-layernorm her zaman eğitilebilir olsun
        if hasattr(vision_model, 'post_layernorm'):
            for p in vision_model.post_layernorm.parameters():
                p.requires_grad = True
        logger.info(f"🔓 {n}/{len(blocks)} blok açık")
    except AttributeError as e:
        logger.warning(f"⚠️  {e}")


# ============================================================
# 🎯 GeM POOLING
# ============================================================
class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) — Ortalama ve Maksimum arasında öğrenilebilir denge.

    Standart mean pooling: her konumun katkısı eşit
    Standart max pooling: yalnızca en baskın özellik aktarılır
    GeM: p parametresi öğrenilerek en uygun orta nokta bulunur

    Formül: GeM(x) = (mean(x^p))^(1/p)
    - p=1 → mean pooling
    - p→∞ → max pooling
    - p=3 (başlangıç): daha ayırt edici özellikler üretir

    Görsel retrieval ve deepfake tespiti gibi ince grained görevlerde
    mean veya max'a kıyasla daha iyi performans gösterir.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)  # Öğrenilebilir p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, SeqLen, Dim]  →  [Batch, Dim]
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


# ============================================================
# 🧠 MODEL
# ============================================================
class SigLIP2Detector(nn.Module):
    """
    SigLIP2 tabanlı görsel deepfake dedektörü.

    Mimari akışı:
        Görsel (336×336) → SigLIP2 vision_model → [B, L, D] token dizisi
        → Mean Pooling + GeM Pooling + Max Pooling  → [B, D×3]
        → Classifier (768→384→128→1) → Sigmoid → P(gerçek)

    Not: Sonuç P(gerçek) olduğundan 1-P(gerçek) = P(sahte)
    """
    def __init__(self, model_name, freeze_backbone=True, unfreeze_last_n=8,
                 dropout_head=0.30, dropout_mid=0.15, use_grad_ckpt=True,
                 feature_dropout_p=0.2, noise_std=0.02, stochastic_depth_p=0.15):
        super().__init__()
        print(f"📥 Backbone yükleniyor...")
        # Yalnızca vision encoder kısmı alınır (text encoder dahil değil)
        self.vision_model    = AutoModel.from_pretrained(model_name).vision_model
        self.noise_std       = noise_std
        self.stoch_p         = stochastic_depth_p
        self.feature_dropout = nn.Dropout(feature_dropout_p)

        # Üçüncü havuzlama kanalı: GeM (v9'da eklendi)
        self.gem = GeM(p=3.0)

        # Backbone dondurma ve seçici açma
        if freeze_backbone:
            for p in self.vision_model.parameters():
                p.requires_grad = False
        unfreeze_last_n_blocks(self.vision_model, unfreeze_last_n)

        # Gradient checkpointing: Aktivasyonları yeniden hesaplayarak VRAM tasarrufu yapar
        if use_grad_ckpt:
            try:
                self.vision_model.gradient_checkpointing_enable()
            except Exception:
                pass

        hidden = self.vision_model.config.hidden_size

        # v9 Classifier: Üç havuzlama kanalı birleştirildiğinden giriş hidden*3
        # hidden*3 → 768 → 384 → 128 → 1
        # LayerNorm + GELU kombinasyonu: daha kararlı eğitim, daha iyi gradyan akışı
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout_head),
            nn.Linear(768, 384),        nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout_mid),
            nn.Linear(384, 128),        nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout_mid * 0.5),
            nn.Linear(128, 1),          # Ham logit — sigmoid dışarıda uygulanır
        )
        # Xavier başlatma: gradyan patlaması/kaybını önler
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Toplam: {total:,}  |  Eğitilebilir: {trainable:,}  |  Frozen: {total-trainable:,}")

    def _stoch(self, x):
        """
        Stochastic Depth (Layer Drop): Eğitimde bir özellik vektörünü
        'stoch_p' olasılıkla sıfırlar. Dropout'a benzer ama vektör bazında.
        """
        if not self.training or self.stoch_p == 0:
            return x
        s = 1 - self.stoch_p
        r = (torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype) + s).floor_()
        return x * r / s  # Ölçek düzeltmesi ile beklenti değeri korunur

    def forward(self, pixel_values):
        # Backbone'dan token dizisi al: [Batch, NumTokens, HiddenDim]
        out  = self.vision_model(pixel_values=pixel_values,
                                  output_hidden_states=False,
                                  interpolate_pos_encoding=True)
        h    = out.last_hidden_state  # [B, L, D]

        # Üç farklı havuzlama stratejisi
        mean = h.mean(1)          # [B, D] — Tüm token'ların ortalaması
        gem  = self.gem(h)        # [B, D] — Genelleştirilmiş ortalama (öğrenilebilir)
        mx   = h.max(1).values    # [B, D] — Her boyuttaki maksimum değer

        # Eğitimde: gürültü + stochastic depth ile regularization
        if self.training:
            mean = mean + torch.randn_like(mean) * self.noise_std
            gem  = gem  + torch.randn_like(gem)  * self.noise_std
            mx   = mx   + torch.randn_like(mx)   * self.noise_std
            mean = self._stoch(mean)
            gem  = self._stoch(gem)
            mx   = self._stoch(mx)

        # Feature dropout
        mean = self.feature_dropout(mean)
        gem  = self.feature_dropout(gem)
        mx   = self.feature_dropout(mx)

        # Üç kanalı birleştir → hidden*3 boyutlu özellik vektörü
        return self.classifier(torch.cat([mean, gem, mx], dim=1)).squeeze(1)


# ============================================================
# 🎲 MIXUP / CUTMIX
# ============================================================
def mixup_batch(images, labels, alpha=0.3):
    """
    Mixup: İki görseli ve etiketlerini Lambda ile doğrusal harmanlar.
    Lambda ~ Beta(alpha, alpha)
    Daha yüksek alpha → daha güçlü harmanlama → daha zorlu eğitim
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[idx], lam * labels + (1 - lam) * labels[idx]


def cutmix_batch(images, labels):
    """
    CutMix: Mixup'tan farklı olarak görselin dikdörtgen bir bölgesi
    başka bir görselin karesiyle değiştirilir.
    Etiket, kesilen alanın oranına göre karıştırılır.
    Yerel bölgesel özelliklerin öğrenilmesini teşvik eder.
    """
    lam  = float(np.random.beta(1.0, 1.0))
    idx  = torch.randperm(images.size(0), device=images.device)
    H, W = images.size(2), images.size(3)
    # Kesim boyutları — lambda ile orantılı alan
    rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
    cy, cx = random.randint(0, H), random.randint(0, W)
    y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
    x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)  # Gerçek lambda (piksel oranı)
    return images, lam * labels + (1 - lam) * labels[idx]


# ============================================================
# 🎯 LOSS FONKSİYONLARI
# ============================================================
def focal_loss(logits, targets, gamma=1.5, alpha=0.50, eps=0.05):
    """
    Focal Loss: Zor örneklere (yüksek kayıp) daha fazla, kolay örneklere
    (düşük kayıp) daha az ağırlık verir.

    Formül: FL = -α × (1-p_t)^γ × log(p_t)
    - gamma: Kolay örneklerin ağırlık azalma hızı (0=standart BCE, yüksek=sert)
    - alpha: Sınıf dengesi faktörü
    - eps: Label smoothing miktarı
    """
    t   = targets * (1 - eps) + 0.5 * eps  # Label smoothing uygula
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction='none')
    p   = torch.sigmoid(logits)
    pt  = (p * targets + (1 - p) * (1 - targets)).clamp(EPSILON, 1 - EPSILON)
    at  = alpha * targets + (1 - alpha) * (1 - targets)  # Sınıf ağırlığı
    return (at * (1 - pt) ** gamma * bce).mean()


def smooth_bce(logits, targets, eps=0.05):
    """Standart Binary Cross-Entropy + Label Smoothing."""
    t = targets * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(logits, t)


def compute_loss(logits, labels, config):
    """CONFIG'e göre uygun kayıp fonksiyonunu çağırır."""
    if config.get("use_focal_loss", True):
        return focal_loss(logits, labels,
                          gamma=config.get("focal_gamma", 1.5),
                          alpha=config.get("focal_alpha", 0.50),
                          eps=config.get("label_smoothing", 0.05))
    return smooth_bce(logits, labels, eps=config.get("label_smoothing", 0.05))


# ============================================================
# 🔓 PROGRESSİVE UNFREEZE
# ============================================================
def apply_progressive_unfreeze(model, epoch, config):
    """
    Her epoch başında CONFIG'deki schedule'a göre uygun sayıda bloğu açar.
    Örnek: 0. epoch'ta 8 blok, 8. epoch'ta 16 blok açılır.
    """
    if not config.get("progressive_unfreeze", False):
        return
    sched = config.get("progressive_unfreeze_schedule", {})
    n = None
    # Mevcut epoch için en yakın (küçük) schedule anahtarını bul
    for k in sorted(sched.keys(), reverse=True):
        if epoch >= k:
            n = sched[k]
            break
    if n is not None:
        unfreeze_last_n_blocks(model.vision_model, n)


# ============================================================
# 🏋️ TEK EPOCH EĞİTİMİ
# ============================================================
def train_one_epoch(model, loader, optimizer, config, epoch, scaler):
    """
    Bir epoch boyunca eğitimi yönetir:
    - Gradient accumulation (efektif batch büyütme)
    - Mixup / CutMix augmentation
    - AMP (FP16) ile hızlandırılmış hesaplama
    - R-Drop (opsiyonel çift forward + KL divergence)
    - Gradient clipping (norm > 1.0 ise kırp)
    """
    model.train()
    device      = config["device"]
    accum       = config.get("grad_accum_steps", 2)   # Gradient biriktirme adım sayısı
    use_amp     = config.get("use_amp", True)
    use_rdrop   = config.get("use_rdrop", False)
    rdrop_alpha = config.get("rdrop_alpha", 0.5)
    use_mixup   = config.get("use_mixup", False)
    mixup_alpha = config.get("mixup_alpha", 0.3)
    cutmix_prob = config.get("cutmix_prob", 0.2)

    total_loss, preds_all, labels_all = 0.0, [], []
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for i, (imgs, lbls) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True, dtype=torch.float32)
        lbls = lbls.to(device, non_blocking=True)

        # %35 olasılıkla Mixup veya CutMix uygula
        if use_mixup and random.random() < 0.35:
            if random.random() < cutmix_prob:
                imgs, lbls = cutmix_batch(imgs, lbls)    # Bölge değiştirme
            else:
                imgs, lbls = mixup_batch(imgs, lbls, mixup_alpha)  # Karıştırma

        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_rdrop:
                # Aynı girdiyi iki farklı dropout maskesiyle işle
                l1, l2 = model(imgs), model(imgs)
                ce = (compute_loss(l1, lbls, config) + compute_loss(l2, lbls, config)) / 2
                p1 = torch.sigmoid(l1).clamp(EPSILON, 1 - EPSILON)
                p2 = torch.sigmoid(l2).clamp(EPSILON, 1 - EPSILON)
                # İki dağılım arasındaki KL divergence regularization olarak eklenir
                kl = ((F.kl_div(p1.log(), p2, reduction='batchmean') +
                       F.kl_div(p2.log(), p1, reduction='batchmean')) / 2)
                loss   = (ce + rdrop_alpha * kl) / accum
                logits = (l1 + l2) / 2
            else:
                logits = model(imgs)
                loss   = compute_loss(logits, lbls, config) / accum  # Gradyan birikimi için böl

        scaler.scale(loss).backward()  # FP16 ölçeklü gradyan hesaplama

        # Her 'accum' adımda bir optimizer adımı at
        if (i + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradyan kırpma
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum  # Gerçek kayıp değerini geri al
        prob = torch.sigmoid(logits).detach().float().cpu().numpy()
        preds_all.extend((prob > 0.5).astype(int))
        labels_all.extend((lbls.cpu().numpy() > 0.5).astype(int))
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")

    # Kalan gradyanları işle (toplam adım sayısı accum'ın katı değilse)
    if len(loader) % accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

    return total_loss / len(loader), accuracy_score(labels_all, preds_all) * 100


# ============================================================
# ✅ DOĞRULAMA
# ============================================================
def validate(model, loader, config, epoch, ema_threshold=None):
    """
    Validation setinde model performansını değerlendirir.
    Sabit 0.5 eşiği yerine dinamik threshold optimizasyonu uygular:
    0.25-0.75 arasında 0.02 adımlarla F1'i maksimize eden eşiği bulur.
    """
    model.eval()
    device  = config["device"]
    use_amp = config.get("use_amp", True)
    total_loss, probs_all, labels_all = 0.0, [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
            imgs = imgs.to(device, non_blocking=True, dtype=torch.float32)
            lbls = lbls.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss   = compute_loss(logits, lbls, config)
            total_loss += loss.item()
            probs_all.extend(torch.sigmoid(logits).float().cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())

    lbl = np.array(labels_all)
    prb = np.array(probs_all)

    # Dinamik threshold arama: F1'i maksimize eden eşiği bul
    best_t, best_f1_t = 0.5, 0.0
    for t in np.arange(0.25, 0.76, 0.02):
        pred_t = (prb > t).astype(int)
        f1_t   = f1_score(lbl, pred_t, average="binary", zero_division=0)
        if f1_t > best_f1_t:
            best_f1_t, best_t = f1_t, t

    # EMA eşik yumuşatma (alpha > 0 ise geçmiş epoch eşiği karıştırılır)
    ema_alpha = config.get("ema_threshold_alpha", 0.0)
    if ema_threshold is not None and ema_alpha > 0.0:
        best_t = ema_alpha * ema_threshold + (1 - ema_alpha) * best_t

    # Seçilen threshold ile metrikleri hesapla
    pred = (prb > best_t).astype(int)
    acc  = accuracy_score(lbl, pred) * 100
    f1   = f1_score(lbl, pred, average="binary", zero_division=0)
    prec = precision_score(lbl, pred, average="binary", zero_division=0)
    rec  = recall_score(lbl, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(lbl, prb)  # Threshold bağımsız — ham olasılıklar kullanılır
    except Exception:
        auc = 0.0

    return total_loss / len(loader), acc, auc, f1, prec, rec, prb, lbl, best_t


# ============================================================
# 🔁 TEST-TIME AUGMENTATION (TTA)
# ============================================================
def predict_with_tta(model, loader, device, n_aug=5, use_amp=True):
    """
    Aynı görseli N farklı augmentation ile modelden geçirir ve
    tahminlerin ortalamasını alır.

    Avantaj: Tek tahmine kıyasla ~%1-3 daha iyi AUC/F1
    Dezavantaj: N kat yavaş inference

    Not: Loader görüntüleri normalizasyonlu tensor olarak içerdiğinden
    önce denormalize edilip augmentation sonrası yeniden normalize edilir.
    """
    model.eval()
    transforms  = get_tta_transforms()[:n_aug]
    mean_np     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_np      = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    acc_probs   = None
    all_labels  = []

    for aug_i, tfm in enumerate(transforms):
        probs = []
        with torch.no_grad():
            for imgs, lbls in loader:
                aug_imgs = []
                for img in imgs.numpy():
                    # Tensor → numpy (denormalize) → augmentation → normalize → tensor
                    hw = np.transpose(img, (1, 2, 0))
                    hw = np.clip(hw * std_np + mean_np, 0, 1)
                    hw = (hw * 255).astype(np.uint8)
                    aug_imgs.append(tfm(image=hw)["image"])
                batch = torch.stack(aug_imgs).to(device, dtype=torch.float32)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(batch)
                probs.extend(torch.sigmoid(logits).float().cpu().numpy())
                if aug_i == 0:
                    all_labels.extend(lbls.numpy())
        arr       = np.array(probs)
        acc_probs = arr if acc_probs is None else acc_probs + arr  # Birikimli toplam

    return acc_probs / len(transforms), np.array(all_labels)  # Ortalama


# ============================================================
# 🎯 THRESHOLD OPTİMİZASYONU
# ============================================================
def find_best_threshold(probs, labels, metric="f1"):
    """
    Test seti üzerinde F1 veya accuracy'yi maksimize eden
    karar eşiğini ızgara aramasıyla (0.01 adım) bulur.
    Bu eşik model dosyasına kaydedilir ve inference sırasında kullanılır.
    """
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.25, 0.76, 0.01):
        pred = (probs > t).astype(int)
        s    = f1_score(labels, pred, zero_division=0) if metric == "f1" \
               else accuracy_score(labels, pred)
        if s > best_s:
            best_s, best_t = s, t
    print(f"🎯 Optimal threshold: {best_t:.2f}  ({metric}={best_s:.4f})")
    return float(best_t)


# ============================================================
# 📊 GÖRSELLEŞTİRME
# ============================================================
def plot_epoch_snapshot(history, epoch, config):
    """Her epoch sonunda 4 grafik içeren anlık görüntü kaydeder."""
    ep = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SigLIP2 v9 — Epoch {epoch+1}', fontsize=15, fontweight='bold')

    # Kayıp eğrisi
    axes[0,0].plot(ep, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0,0].plot(ep, history["val_loss"],   "r-o", ms=4, label="Val")
    axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    # Accuracy eğrisi (gölgeli alan = train-val farkı)
    axes[0,1].plot(ep, history["train_acc"], "b-o", ms=4, label="Train")
    axes[0,1].plot(ep, history["val_acc"],   "r-o", ms=4, label="Val")
    axes[0,1].fill_between(ep, history["train_acc"], history["val_acc"], alpha=0.15, color="gray")
    axes[0,1].set_title("Accuracy (%)"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    # F1, AUC, Precision, Recall
    axes[1,0].plot(ep, history["val_f1"],  "g-o", ms=4, label="F1")
    axes[1,0].plot(ep, history["val_auc"], "k-o", ms=4, label="AUC")
    if "val_precision" in history and history["val_precision"]:
        axes[1,0].plot(ep, history["val_precision"], "b--o", ms=3, label="Precision")
        axes[1,0].plot(ep, history["val_recall"],    "r--o", ms=3, label="Recall")
    axes[1,0].set_title("Val Metrics"); axes[1,0].set_ylim(0, 1.05)
    axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    # Dinamik threshold veya overfitting gap
    if "threshold_history" in history and history["threshold_history"]:
        axes[1,1].plot(ep, history["threshold_history"], "m-o", ms=4, label="Dynamic Threshold")
        axes[1,1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5")
        axes[1,1].set_title("Dynamic Threshold"); axes[1,1].set_ylim(0.2, 0.8)
        axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
    else:
        overfit = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
        colors  = ["green" if x < 5 else "orange" if x < 10 else "red" for x in overfit]
        axes[1,1].bar(ep, overfit, color=colors, alpha=0.75)
        axes[1,1].axhline(5,  color="orange", linestyle="--", label="Uyarı (>5%)")
        axes[1,1].axhline(10, color="red",    linestyle="--", label="Overfit (>10%)")
        axes[1,1].set_title("Overfit Gap (Train-Val Acc)"); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config["local_epoch_plot_dir"], f"epoch_{epoch+1:03d}.png"),
                dpi=80, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(tp, fp, tn, fn, path, threshold=0.5, title=""):
    """Confusion matrix görselleştirir ve metrik özetini yanına ekler."""
    try:
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap=plt.cm.Blues)
        thresh = cm.max() / 2
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if v > thresh else "black")
        ax.set(xticks=[0,1], yticks=[0,1],
               xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
               title=title, ylabel="Gerçek Etiket", xlabel="Tahmin")
        total = tp+tn+fp+fn
        acc   = (tp+tn)/(total+EPSILON)*100
        prec  = tp/(tp+fp+EPSILON)
        rec   = tp/(tp+fn+EPSILON)
        f1v   = 2*prec*rec/(prec+rec+EPSILON)
        ax.text(1.3, 0.5,
                f"t={threshold:.2f}\nAcc:{acc:.1f}%\nPrec:{prec:.3f}\nRec:{rec:.3f}\nF1:{f1v:.3f}",
                transform=ax.transAxes, fontsize=10, va="center", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow"))
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"⚠️  plot_cm: {e}")


def plot_history(history, config):
    """Eğitim tamamlandığında özet grafik kaydeder."""
    try:
        ep   = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Training History — SigLIP2 v9", fontsize=13, fontweight="bold")
        axes[0].plot(ep, history["train_loss"], label="Train"); axes[0].plot(ep, history["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep, history["train_acc"],  label="Train"); axes[1].plot(ep, history["val_acc"],  label="Val")
        axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(ep, history["val_f1"],  label="F1")
        axes[2].plot(ep, history["val_auc"], label="AUC")
        axes[2].set_title("Val F1 & AUC"); axes[2].legend(); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config["project_dir"], "training_history_v9.png"), dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"⚠️  plot_history: {e}")


# ============================================================
# 🚀 ANA EĞİTİM FONKSİYONU
# ============================================================
def train(config: dict):
    """
    SigLIP2 v9 eğitim döngüsü.

    Adımlar:
    1. Dataset kopyalama (Drive → SSD)
    2. Dataset ve DataLoader oluşturma
    3. Model, optimizer, scheduler başlatma
    4. Her epoch: train → validate → progressive unfreeze → SWA güncelleme
    5. En iyi modeli kaydet (F1 + AUC eşiği)
    6. Test seti değerlendirmesi + TTA
    """
    device = config["device"]
    print(f"\n{'='*60}")
    print(f"🚀 SigLIP2 v9 — F1>0.90 Hedefli")
    print(f"   Device: {device}  |  AMP: {config['use_amp']}")
    print(f"   Efektif batch: {config['batch_size'] * config['grad_accum_steps']}")
    print(f"   Image size: {config['image_size']}px")
    print(f"   Backbone başlangıç açık blok: {config['unfreeze_last_n']}")
    print(f"   Backbone LR: {config['learning_rate'] * 0.25:.1e}  (çarpan 0.25)")
    print(f"   Focal Loss: {config['use_focal_loss']}  |  focal_alpha: {config['focal_alpha']}")
    print(f"   Early stopping patience: {config['early_stopping_patience']}")
    print(f"   SWA start epoch: {config['swa_start_epoch']}")
    print(f"   Pooling: mean + GeM + max → hidden*3")
    print(f"   Classifier: hidden*3 → 768 → 384 → 128 → 1")
    print(f"{'='*60}\n")

    copy_dataset_to_ssd(DRIVE_DATASET, LOCAL_DATASET)

    train_ds, val_ds, test_ds = build_datasets(config)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"\n📊 Toplam: {total} görsel\n")

    nw = config["num_workers"]
    kw = dict(
        batch_size         = config["batch_size"],
        num_workers        = nw,
        pin_memory         = True,           # GPU'ya aktarımı hızlandırır
        persistent_workers = nw > 0,         # Worker süreçlerini epoch aralarında canlı tut
        prefetch_factor    = 4 if nw > 0 else None,  # Paralel veri önyüklemesi
    )

    # Sınıf dengeli örnekleyici ile eğitim loader'ı
    if config.get("use_balanced_sampler", True):
        sampler      = make_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True,  **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)

    val_loader  = DataLoader(val_ds,  shuffle=False, drop_last=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **kw)

    # Model başlatma
    model = SigLIP2Detector(
        model_name         = config["model_name"],
        freeze_backbone    = config["freeze_backbone"],
        unfreeze_last_n    = config["unfreeze_last_n"],
        dropout_head       = config["dropout_head"],
        dropout_mid        = config["dropout_mid"],
        use_grad_ckpt      = config["use_grad_ckpt"],
        feature_dropout_p  = config["feature_dropout_p"],
        noise_std          = config["noise_std"],
        stochastic_depth_p = config["stochastic_depth_prob"],
    ).to(device)

    if config.get("use_compile", False):
        try:
            model = torch.compile(model)  # PyTorch 2.0 graph compilation
            print("⚡ torch.compile aktif")
        except Exception:
            pass

    # FP16 gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])

    # Optimizer: backbone ve classifier için ayrı LR
    bb_params  = [p for p in model.vision_model.parameters()  if p.requires_grad]
    clf_params = list(model.gem.parameters()) + list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": bb_params,  "lr": config["learning_rate"] * 0.25, "name": "backbone"},   # 5e-5
        {"params": clf_params, "lr": config["learning_rate"],         "name": "classifier"}, # 2e-4
    ], weight_decay=config["weight_decay"])

    # Scheduler: Isınma + Cosine Annealing Warm Restarts
    warmup = config["warmup_epochs"]
    warmup_sched = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: (e+1)/max(warmup, 1) if e < warmup else 1.0)
    if config.get("use_cosine_restarts"):
        cosine_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config["cosine_T0"], T_mult=config["cosine_T_mult"], eta_min=1e-7)
    else:
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config["epochs"] - warmup, 1), eta_min=1e-7)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], [warmup])

    # Stochastic Weight Averaging (SWA): Birden fazla checkpoint'ı ortalamalayarak
    # daha düzgün loss yüzeyi bulur ve genelleştirmeyi artırır
    use_swa = config.get("use_swa", True)
    swa_model = swa_sched = None
    if use_swa:
        swa_model = optim.swa_utils.AveragedModel(model)
        swa_sched = optim.swa_utils.SWALR(optimizer, swa_lr=config["swa_lr"])

    early_stop = EarlyStopping(config["early_stopping_patience"], config["early_stopping_min_delta"])

    # Metrik geçmişi
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "val_auc": [], "val_f1": [], "val_precision": [], "val_recall": [],
        "best_threshold": 0.5, "threshold_history": [],
    }
    best_f1       = 0.0
    best_thr      = 0.5
    ema_threshold = None

    # ── Eğitim Döngüsü ──────────────────────────────────────
    for epoch in range(config["epochs"]):
        # Her epoch başında progressive unfreeze schedule'ını uygula
        apply_progressive_unfreeze(model, epoch, config)

        # Yeni açılan parametreleri optimizer'a ekle
        bb_params_now = [p for p in model.vision_model.parameters() if p.requires_grad]
        optimizer.param_groups[0]["params"] = bb_params_now

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, config, epoch, scaler)

        val_loss, val_acc, val_auc, val_f1, val_prec, val_rec, val_probs, val_lbl, dynamic_thr = validate(
            model, val_loader, config, epoch, ema_threshold=ema_threshold)

        ema_threshold = dynamic_thr  # Bir sonraki epoch için EMA threshold güncelle

        # SWA aktifse SWA scheduler kullan, değilse normal scheduler
        if use_swa and epoch >= config["swa_start_epoch"]:
            swa_model.update_parameters(model)  # SWA ağırlık ortalamasına ekle
            swa_sched.step()
        else:
            scheduler.step()

        # Geçmiş güncelle
        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("val_loss", val_loss),   ("val_acc", val_acc),
                     ("val_auc", val_auc),      ("val_f1", val_f1),
                     ("val_precision", val_prec), ("val_recall", val_rec)]:
            history[k].append(v)
        history["threshold_history"].append(dynamic_thr)

        # Overfitting gap
        gap = tr_acc - val_acc
        tag = "🟢" if gap < 5 else "🟡" if gap < 10 else "🔴 OVERFIT"
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1}/{config['epochs']}  ({elapsed:.0f}s)")
        print(f"  Train  Loss:{tr_loss:.4f}  Acc:{tr_acc:.2f}%")
        print(f"  Val    Loss:{val_loss:.4f}  Acc:{val_acc:.2f}%  AUC:{val_auc:.4f}  F1:{val_f1:.4f}")
        print(f"  Prec:{val_prec:.4f}  Rec:{val_rec:.4f}  Threshold:{dynamic_thr:.3f}  Gap:{gap:.2f}%  {tag}")

        plot_epoch_snapshot(history, epoch, config)

        # En iyi model kaydı — hem F1 hem AUC eşiğini geçmeli
        if val_f1 > best_f1 and val_auc > config["min_auc_to_save"]:
            best_f1  = val_f1
            best_thr = find_best_threshold(val_probs, val_lbl)  # Test için optimal threshold
            history["best_threshold"] = best_thr

            if os.path.exists(config["save_path"]):
                os.remove(config["save_path"])
            torch.save({
                "epoch": epoch+1, "model_state_dict": model.state_dict(),
                "best_f1": best_f1, "best_threshold": best_thr,
                "val_auc": val_auc, "val_acc": val_acc, "config": config,
            }, config["save_path"])
            print(f"  💾 Kaydedildi! F1:{best_f1:.4f}  AUC:{val_auc:.4f}  t={best_thr:.2f}")

        if early_stop(val_auc, val_f1):
            print(f"\n⏹️  Early stopping — epoch {epoch+1}")
            break

    # SWA modeli için Batch Normalization istatistiklerini güncelle
    if use_swa and swa_model is not None:
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            swa_path = config["save_path"].replace(".pth", "_swa.pth")
            torch.save({"model_state_dict": swa_model.module.state_dict(),
                        "best_threshold": best_thr, "config": config}, swa_path)
            print(f"💾 SWA modeli kaydedildi: {swa_path}")
        except Exception as e:
            logger.warning(f"SWA BN: {e}")

    plot_history(history, config)

    # ── Test Değerlendirmesi ──────────────────────────────────
    print(f"\n{'='*60}\n🧪 TEST DEĞERLENDİRMESİ\n{'='*60}")

    if not os.path.exists(config["save_path"]):
        print("⚠️  Model kaydedilmedi.")
        return model, history, {}

    # En iyi checkpoint'ı yükle
    ckpt = torch.load(config["save_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    thr = ckpt.get("best_threshold", 0.5)
    print(f"   Epoch {ckpt['epoch']}  F1={ckpt['best_f1']:.4f}  AUC={ckpt['val_auc']:.4f}  t={thr:.2f}")

    model.eval()
    t_probs, t_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Test", leave=False):
            imgs = imgs.to(device, non_blocking=True, dtype=torch.float32)
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                probs = torch.sigmoid(model(imgs)).float().cpu().numpy()
            t_probs.extend(probs)
            t_labels.extend(lbls.numpy().astype(int))

    t_lbl  = np.array(t_labels)
    t_prb  = np.array(t_probs)
    t_pred = (t_prb > best_thr).astype(int)
    t_f1   = f1_score(t_lbl, t_pred, average="binary", zero_division=0)
    t_acc  = accuracy_score(t_lbl, t_pred) * 100
    t_prec = precision_score(t_lbl, t_pred, average="binary", zero_division=0)
    t_rec  = recall_score(t_lbl, t_pred, average="binary", zero_division=0)
    try:
        t_auc = roc_auc_score(t_lbl, t_prb)
    except Exception:
        t_auc = 0.0

    # Val → Test genelleştirme boşluğu
    gap_gen = abs(best_f1 - t_f1)
    gen_tag = "🟢 İyi" if gap_gen < 0.03 else "🟡 Kabul" if gap_gen < 0.07 else "🔴 Sorun"

    print(f"\n   Test  Acc:{t_acc:.2f}%  F1:{t_f1:.4f}  AUC:{t_auc:.4f}")
    print(f"   Prec:{t_prec:.4f}  Rec:{t_rec:.4f}")
    print(f"   Val-Test F1 farkı: {gap_gen:.4f}  {gen_tag}")

    # Confusion matrix
    tp = int(((t_pred==1)&(t_lbl==1)).sum()); fp = int(((t_pred==1)&(t_lbl==0)).sum())
    tn = int(((t_pred==0)&(t_lbl==0)).sum()); fn = int(((t_pred==0)&(t_lbl==1)).sum())
    plot_confusion_matrix(tp, fp, tn, fn,
                          os.path.join(config["project_dir"], "confusion_matrix_test_v9.png"),
                          thr, f"TEST t={thr:.2f}")

    # TTA ile son değerlendirme
    if config.get("use_tta", True):
        print("\n🔁 TTA değerlendirmesi...")
        try:
            tta_prb, tta_lbl = predict_with_tta(model, test_loader, device,
                                                 config["tta_n_aug"], config["use_amp"])
            tta_pred = (tta_prb > thr).astype(int)
            tta_f1   = f1_score(tta_lbl, tta_pred, average="binary", zero_division=0)
            tta_auc  = roc_auc_score(tta_lbl, tta_prb) if len(np.unique(tta_lbl)) > 1 else 0.0
            print(f"   TTA F1:{tta_f1:.4f}  AUC:{tta_auc:.4f}")
        except Exception as e:
            logger.warning(f"TTA: {e}")

    results = {"test_acc": t_acc, "test_f1": t_f1, "test_auc": t_auc,
               "test_precision": t_prec, "test_recall": t_rec, "best_threshold": thr}
    ckpt.update(results)
    torch.save(ckpt, config["save_path"])  # Test sonuçlarını da checkpoint'a ekle

    print(f"\n💾 Çıktılar → {config['project_dir']}")
    print("✅ Eğitim tamamlandı.")
    return model, history, results


# ============================================================
# ▶️  BAŞLAT
# ============================================================
if __name__ == "__main__":
    model, history, results = train(CONFIG)
