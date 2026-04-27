# -*- coding: utf-8 -*-


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
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

EPSILON = 1e-7

# ============================================================
# ⚙️  KONFİGÜRASYON
# ============================================================
DRIVE_DATASET = "/content/drive/MyDrive/dataset"
LOCAL_DATASET = "/content/local_dataset"
PROJECT_DIR   = "project_dir"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "epoch_plots"), exist_ok=True)

CONFIG = {
    # ── Model ──────────────────────────────────────────────
    "model_name":        "google/siglip2-so400m-patch14-384",
    "epochs":            50,
    "batch_size":        32,
    "grad_accum_steps":  1,
    "use_amp":           False,
    "use_grad_ckpt":     True,
    "learning_rate":     1e-4,
    "weight_decay":      0.05,
    "image_size":        224,

    # ── Dataset — SSD yolları ──────────────────────────────
    "train_real_dir": os.path.join(LOCAL_DATASET, "train", "real"),
    "train_fake_dir": os.path.join(LOCAL_DATASET, "train", "fake"),
    "val_real_dir":   os.path.join(LOCAL_DATASET, "val",   "real"),
    "val_fake_dir":   os.path.join(LOCAL_DATASET, "val",   "fake"),
    "test_real_dir":  os.path.join(LOCAL_DATASET, "test",  "real"),
    "test_fake_dir":  os.path.join(LOCAL_DATASET, "test",  "fake"),
    "test_split_ratio":  0.15,
    "val_split_ratio":   0.15,

    # ── Kayıt ──────────────────────────────────────────────
    "save_path":            os.path.join(PROJECT_DIR, "best_model.pth"),
    "local_epoch_plot_dir": os.path.join(PROJECT_DIR, "epoch_plots"),
    "project_dir":          PROJECT_DIR,

    # ── DataLoader ─────────────────────────────────────────
    "num_workers":  4,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "cache_images": False,

    # ── Backbone ───────────────────────────────────────────
    "freeze_backbone":    True,
    "unfreeze_last_n":    0,
    "progressive_unfreeze": True,
    "progressive_unfreeze_schedule": {
        0:  4,
        5:  6,
        12: 8,
        20: 10,
        30: 12,
    },

    # ── Augmentation ───────────────────────────────────────
    "use_distortion_aug": True,
    "use_mixup":          False,
    "mixup_alpha":        0.4,
    "cutmix_prob":        0.3,

    # ── Regularization ─────────────────────────────────────
    "early_stopping_patience":  10,
    "early_stopping_min_delta": 5e-4,
    "label_smoothing":          0.05,
    "dropout_head":             0.35,
    "dropout_mid":              0.20,
    "warmup_epochs":            2,
    "stochastic_depth_prob":    0.10,

    # ── Loss ───────────────────────────────────────────────
    "use_focal_loss": True,
    "focal_gamma":    2.0,
    "focal_alpha":    0.45,

    # ── TTA ────────────────────────────────────────────────
    "use_tta":   True,
    "tta_n_aug": 5,

    # ── R-Drop ─────────────────────────────────────────────
    "use_rdrop":   False,
    "rdrop_alpha": 0.5,

    # ── Fusion ─────────────────────────────────────────────
    "fusion_type":        "simple",
    "feature_dropout_p":  0.15,

    # ── Scheduler ──────────────────────────────────────────
    "use_cosine_restarts": True,
    "cosine_T0":           7,
    "cosine_T_mult":       2,

    # ── Noise ──────────────────────────────────────────────
    "noise_std": 0.02,

    # ── SWA ────────────────────────────────────────────────
    "use_swa":         True,
    "swa_start_epoch": 18,
    "swa_lr":          1e-6,

    # ── Balanced sampler ───────────────────────────────────
    "use_balanced_sampler": True,

    # ── Model kayıt eşiği ──────────────────────────────────
    "min_auc_to_save": 0.75,

    # ── PyTorch compile ────────────────────────────────────
    "use_compile": False,
}


# ============================================================
# 🚀 DRIVE → SSD KOPYALAMA
# ============================================================
def copy_dataset_to_ssd(drive_root: str, local_root: str, force: bool = False):
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
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        if random.random() < 0.5:
            grid = np.zeros_like(img, dtype=np.float32)
            for i in range(0, h, 8):
                grid[i:min(i+1, h), :] += random.uniform(1, 5)
            img = np.clip(img.astype(np.float32) + grid * random.uniform(0.1, 0.4), 0, 255).astype(np.uint8)
        return img

    def get_transform_init_args_names(self):
        return ()


FAST_DISTORTION_POOL = [
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    A.ImageCompression(quality_lower=40, quality_upper=85, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    A.RandomGamma(gamma_limit=(70, 130), p=1.0),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    A.CoarseDropout(max_holes=6, max_height=32, max_width=32, min_holes=1, p=1.0),
    A.HorizontalFlip(p=1.0),
    A.Rotate(limit=10, p=1.0),
    FrequencyAugment(p=1.0),
]

_NORM = [
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
]

def get_train_transform(image_size: int = 224):
    resize = [A.Resize(image_size, image_size)]
    level  = random.choices([1, 2, 3], weights=[0.40, 0.45, 0.15])[0]
    if level == 1:
        aug = []
    elif level == 2:
        aug = random.sample(FAST_DISTORTION_POOL, random.randint(1, 3))
    else:
        aug = random.sample(FAST_DISTORTION_POOL, random.randint(3, 5))
    return A.Compose(resize + aug + _NORM)


VAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def get_tta_transforms() -> List[A.Compose]:
    norm = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
    return [
        A.Compose([A.Resize(224, 224)] + norm),
        A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=1.0)] + norm),
        A.Compose([A.Resize(224, 224), A.RandomBrightnessContrast(0.1, 0.1, p=1.0)] + norm),
        A.Compose([A.Resize(224, 224), A.GaussianBlur(blur_limit=(3, 3), p=1.0)] + norm),
        A.Compose([A.Resize(224, 224), A.Rotate(limit=5, p=1.0)] + norm),
    ]


# ============================================================
# 📁 DATASET
# ============================================================
class AIDetectionDataset(Dataset):
    VALID_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    def __init__(self, real_dir, fake_dir, split="train", config=None, samples=None):
        self.config   = config
        self.split    = split
        self.is_train = (split == "train")
        self.cache    = {}

        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            for directory, label in [(real_dir, 1), (fake_dir, 0)]:
                if directory and os.path.exists(directory):
                    for f in sorted(os.listdir(directory)):
                        if f.lower().endswith(self.VALID_EXT):
                            self.samples.append((os.path.join(directory, f), label))
                else:
                    logger.warning(f"⚠️  Klasör bulunamadı: {directory}")
            if self.is_train:
                random.shuffle(self.samples)

        real_c = sum(1 for _, l in self.samples if l == 1)
        fake_c = sum(1 for _, l in self.samples if l == 0)
        print(f"{split.upper():5}: {len(self.samples):>5} görsel  |  Real:{real_c}  Fake:{fake_c}")

        if config and config.get("cache_images", False):
            print(f"   💾 RAM'e yükleniyor ({split})...")
            for path, _ in tqdm(self.samples, leave=False):
                try:
                    self.cache[path] = np.array(Image.open(path).convert("RGB"))
                except Exception:
                    sz = config.get("image_size", 224)
                    self.cache[path] = np.zeros((sz, sz, 3), dtype=np.uint8)
            print(f"   ✅ {len(self.cache)} görsel RAM'de")

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [label for _, label in self.samples]

    def _load(self, path):
        if path in self.cache:
            return self.cache[path]
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            sz = self.config.get("image_size", 224)
            return np.zeros((sz, sz, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self._load(path)
        t = get_train_transform(self.config.get("image_size", 224)) if self.is_train else VAL_TRANSFORM
        return t(image=image)["image"], torch.tensor(label, dtype=torch.float32)


# ============================================================
# ⚖️  BALANCED SAMPLER
# ============================================================
def make_balanced_sampler(dataset):
    labels  = dataset.get_labels()
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(dataset),
        replacement=True,
    )


# ============================================================
# ✅ DÜZELTME 1: STRATIFIED SOURCE SPLIT
# Her kaynaktan orantılı böler — val artık train'den "kolay" olmaz
# ============================================================
def stratified_source_split(all_samples, val_ratio=0.15, test_ratio=0.15):
    groups = defaultdict(list)
    for path, label in all_samples:
        groups[os.path.dirname(path)].append((path, label))

    train, val, test = [], [], []
    for gid, samples in groups.items():
        n = len(samples)
        if n < 3:
            train.extend(samples)
            continue
        # Her gruptan orantılı böl
        tr, temp = train_test_split(samples, test_size=val_ratio + test_ratio, random_state=42)
        v_ratio_adjusted = test_ratio / (val_ratio + test_ratio)
        v, te = train_test_split(temp, test_size=v_ratio_adjusted, random_state=42)
        train.extend(tr)
        val.extend(v)
        test.extend(te)

    random.shuffle(train)
    print(f"   Stratified split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test


def build_datasets(config):
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

    logger.info("📂 Mod B: stratified source split")
    all_s = []
    for d, lbl in [(config["train_real_dir"], 1), (config["train_fake_dir"], 0),
                   (config["val_real_dir"],   1), (config["val_fake_dir"],   0)]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(AIDetectionDataset.VALID_EXT):
                    all_s.append((os.path.join(d, f), lbl))


    tr_s, va_s, te_s = stratified_source_split(
        all_s,
        val_ratio=config.get("val_split_ratio", 0.15),
        test_ratio=config.get("test_split_ratio", 0.15),
    )
    return (
        AIDetectionDataset(None, None, "train", config, tr_s),
        AIDetectionDataset(None, None, "val",   config, va_s),
        AIDetectionDataset(None, None, "test",  config, te_s),
    )


# ============================================================
# ⏹️  EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=5e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")
        self.stop      = False

    def __call__(self, auc, f1):
        metric = 1.0 - (0.6 * auc + 0.4 * f1)
        if metric < self.best - self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ============================================================
# 🧠 MODEL
# ============================================================
def unfreeze_last_n_blocks(vision_model, n):
    if n <= 0:
        for p in vision_model.parameters():
            p.requires_grad = False
        return
    try:
        for p in vision_model.parameters():
            p.requires_grad = False
        blocks = list(vision_model.encoder.layers)
        for block in blocks[max(0, len(blocks) - n):]:
            for p in block.parameters():
                p.requires_grad = True
        if hasattr(vision_model, 'post_layernorm'):
            for p in vision_model.post_layernorm.parameters():
                p.requires_grad = True
        logger.info(f"🔓 {n}/{len(blocks)} blok açık")
    except AttributeError as e:
        logger.warning(f"⚠️  {e}")


class SigLIP2Detector(nn.Module):
    def __init__(self, model_name, freeze_backbone=True, unfreeze_last_n=0,
                 dropout_head=0.35, dropout_mid=0.20, use_grad_ckpt=True,
                 feature_dropout_p=0.15, noise_std=0.02, stochastic_depth_p=0.10):
        super().__init__()
        print(f"📥 Backbone yükleniyor...")
        self.vision_model    = AutoModel.from_pretrained(model_name).vision_model
        self.noise_std       = noise_std
        self.stoch_p         = stochastic_depth_p
        self.feature_dropout = nn.Dropout(feature_dropout_p)

        # Başta tümünü dondur
        for p in self.vision_model.parameters():
            p.requires_grad = False


        if unfreeze_last_n > 0:
            unfreeze_last_n_blocks(self.vision_model, unfreeze_last_n)

        if use_grad_ckpt:
            try:
                self.vision_model.gradient_checkpointing_enable()
            except Exception:
                pass

        hidden = self.vision_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout_head),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout_mid),
            nn.Linear(256, 64),         nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout_mid * 0.5),
            nn.Linear(64, 1),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Toplam: {total:,}  |  Eğitilebilir: {trainable:,}  |  Frozen: {total-trainable:,}")

    def _stoch(self, x):
        if not self.training or self.stoch_p == 0:
            return x
        s = 1 - self.stoch_p
        r = (torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype) + s).floor_()
        return x * r / s

    def forward(self, pixel_values):
        out  = self.vision_model(pixel_values=pixel_values,
                                  output_hidden_states=False,
                                  interpolate_pos_encoding=True)
        h    = out.last_hidden_state
        mean = h.mean(1)
        mx   = h.max(1).values

        if self.training:
            mean = mean + torch.randn_like(mean) * self.noise_std
            mx   = mx   + torch.randn_like(mx)   * self.noise_std
            mean = self._stoch(mean)
            mx   = self._stoch(mx)

        mean = self.feature_dropout(mean)
        mx   = self.feature_dropout(mx)
        return self.classifier(torch.cat([mean, mx], dim=1)).squeeze(1)


# ============================================================
# 🎲 MIXUP / CUTMIX
# ============================================================
def mixup_batch(images, labels, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[idx], lam * labels + (1 - lam) * labels[idx]


def cutmix_batch(images, labels):
    lam  = float(np.random.beta(1.0, 1.0))
    idx  = torch.randperm(images.size(0), device=images.device)
    H, W = images.size(2), images.size(3)
    rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
    cy, cx = random.randint(0, H), random.randint(0, W)
    y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
    x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return images, lam * labels + (1 - lam) * labels[idx]


# ============================================================
# 🎯 LOSS
# ============================================================
def focal_loss(logits, targets, gamma=2.0, alpha=0.45, eps=0.05):
    t   = targets * (1 - eps) + 0.5 * eps
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction='none')
    p   = torch.sigmoid(logits)
    pt  = (p * targets + (1 - p) * (1 - targets)).clamp(EPSILON, 1 - EPSILON)
    at  = alpha * targets + (1 - alpha) * (1 - targets)
    return (at * (1 - pt) ** gamma * bce).mean()


def smooth_bce(logits, targets, eps=0.05):
    t = targets * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(logits, t)


def compute_loss(logits, labels, config):
    if config.get("use_focal_loss", True):
        return focal_loss(logits, labels,
                          gamma=config.get("focal_gamma", 2.0),
                          alpha=config.get("focal_alpha", 0.45),
                          eps=config.get("label_smoothing", 0.05))
    return smooth_bce(logits, labels, eps=config.get("label_smoothing", 0.05))


# ============================================================
# Artık unfreeze olan parametreler optimizer'a da ekleniyor
# ============================================================
def apply_progressive_unfreeze(model, epoch, config, optimizer):
    if not config.get("progressive_unfreeze", False):
        return

    sched = config.get("progressive_unfreeze_schedule", {})
    n = None
    for k in sorted(sched.keys(), reverse=True):
        if epoch >= k:
            n = sched[k]
            break
    if n is None:
        return

    # Backbone bloklarını güncelle
    unfreeze_last_n_blocks(model.vision_model, n)

    # ✅ KRİTİK: Optimizer'ın backbone param_group'unu güncelle
    new_bb_params = [p for p in model.vision_model.parameters() if p.requires_grad]
    if new_bb_params:
        optimizer.param_groups[0]['params'] = new_bb_params
        # Unfreeze oranına göre LR'ı kademeli aç (max 12 blok varsayımı)
        max_blocks = 32  # SigLIP2-so400m blok sayısı
        lr_scale   = 0.01 + 0.04 * (n / max_blocks)  # 0.01x → 0.05x arasında
        optimizer.param_groups[0]['lr'] = config['learning_rate'] * lr_scale
        trainable = sum(p.numel() for p in new_bb_params)
        logger.info(f"   Optimizer güncellendi: {trainable:,} backbone param, LR_scale={lr_scale:.3f}")
    else:
        optimizer.param_groups[0]['params'] = []


# ============================================================
# 🏋️ TRAIN ONE EPOCH
# ============================================================
def train_one_epoch(model, loader, optimizer, config, epoch, scaler):
    model.train()
    device      = config["device"]
    accum       = config.get("grad_accum_steps", 1)
    use_amp     = config.get("use_amp", False)
    use_rdrop   = config.get("use_rdrop", False)
    rdrop_alpha = config.get("rdrop_alpha", 0.5)
    use_mixup   = config.get("use_mixup", False)
    mixup_alpha = config.get("mixup_alpha", 0.4)
    cutmix_prob = config.get("cutmix_prob", 0.3)

    total_loss, preds_all, labels_all = 0.0, [], []
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for i, (imgs, lbls) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True, dtype=torch.float32)
        lbls = lbls.to(device, non_blocking=True)

        if use_mixup and random.random() < 0.35:
            if random.random() < cutmix_prob:
                imgs, lbls = cutmix_batch(imgs, lbls)
            else:
                imgs, lbls = mixup_batch(imgs, lbls, mixup_alpha)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_rdrop:
                l1, l2 = model(imgs), model(imgs)
                ce = (compute_loss(l1, lbls, config) + compute_loss(l2, lbls, config)) / 2
                p1 = torch.sigmoid(l1).clamp(EPSILON, 1 - EPSILON)
                p2 = torch.sigmoid(l2).clamp(EPSILON, 1 - EPSILON)
                kl = ((F.kl_div(p1.log(), p2, reduction='batchmean') +
                       F.kl_div(p2.log(), p1, reduction='batchmean')) / 2)
                loss   = (ce + rdrop_alpha * kl) / accum
                logits = (l1 + l2) / 2
            else:
                logits = model(imgs)
                loss   = compute_loss(logits, lbls, config) / accum

        scaler.scale(loss).backward()

        if (i + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum
        prob = torch.sigmoid(logits).detach().float().cpu().numpy()
        preds_all.extend((prob > 0.5).astype(int))
        labels_all.extend((lbls.cpu().numpy() > 0.5).astype(int))
        pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")

    # Son kalan batch
    if len(loader) % accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / len(loader), accuracy_score(labels_all, preds_all) * 100


# ============================================================
# ✅ VALIDATE
# ============================================================
def validate(model, loader, config, epoch):
    model.eval()
    device  = config["device"]
    use_amp = config.get("use_amp", False)
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

    lbl  = np.array(labels_all)
    prb  = np.array(probs_all)
    pred = (prb > 0.5).astype(int)

    acc  = accuracy_score(lbl, pred) * 100
    f1   = f1_score(lbl, pred, average="binary", zero_division=0)
    prec = precision_score(lbl, pred, average="binary", zero_division=0)
    rec  = recall_score(lbl, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(lbl, prb)
    except Exception:
        auc = 0.0

    return total_loss / len(loader), acc, auc, f1, prec, rec, prb, lbl


# ============================================================
# 🔁 TTA
# ============================================================
def predict_with_tta(model, loader, device, n_aug=5, use_amp=False):
    model.eval()
    transforms = get_tta_transforms()[:n_aug]
    mean_np    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_np     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    acc_probs  = None
    all_labels = []

    for aug_i, tfm in enumerate(transforms):
        probs = []
        with torch.no_grad():
            for imgs, lbls in loader:
                aug_imgs = []
                for img in imgs.numpy():
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
        acc_probs = arr if acc_probs is None else acc_probs + arr

    return acc_probs / len(transforms), np.array(all_labels)


# ============================================================
# 🎯 THRESHOLD OPTİMİZASYONU
# ============================================================
def find_best_threshold(probs, labels, metric="f1"):
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
    ep = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SigLIP2 v8 — Epoch {epoch+1}', fontsize=15, fontweight='bold')

    axes[0,0].plot(ep, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0,0].plot(ep, history["val_loss"],   "r-o", ms=4, label="Val")
    axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(ep, history["train_acc"], "b-o", ms=4, label="Train")
    axes[0,1].plot(ep, history["val_acc"],   "r-o", ms=4, label="Val")
    axes[0,1].fill_between(ep, history["train_acc"], history["val_acc"], alpha=0.15, color="gray")
    axes[0,1].set_title("Accuracy (%)"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(ep, history["val_f1"],  "g-o", ms=4, label="F1")
    axes[1,0].plot(ep, history["val_auc"], "k-o", ms=4, label="AUC")
    axes[1,0].set_title("Val Metrics"); axes[1,0].set_ylim(0, 1.05)
    axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    overfit = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    colors  = ["green" if x < 5 else "orange" if x < 10 else "red" for x in overfit]
    axes[1,1].bar(ep, overfit, color=colors, alpha=0.75)
    axes[1,1].axhline(0,  color="blue",   linestyle="--", linewidth=0.8, label="Sıfır")
    axes[1,1].axhline(5,  color="orange", linestyle="--", label="Uyarı (+5%)")
    axes[1,1].axhline(10, color="red",    linestyle="--", label="Overfit (+10%)")
    axes[1,1].set_title("Overfit Gap (Train - Val)")
    axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config["local_epoch_plot_dir"], f"epoch_{epoch+1:03d}.png"),
                dpi=80, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(tp, fp, tn, fn, path, threshold=0.5, title=""):
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
               title=title, ylabel="Gerçek", xlabel="Tahmin")
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
    try:
        ep = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Training History — SigLIP2 v8", fontsize=13, fontweight="bold")
        axes[0].plot(ep, history["train_loss"], label="Train"); axes[0].plot(ep, history["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep, history["train_acc"],  label="Train"); axes[1].plot(ep, history["val_acc"],  label="Val")
        axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(ep, history["val_f1"],  label="F1")
        axes[2].plot(ep, history["val_auc"], label="AUC")
        axes[2].set_title("Val F1 & AUC"); axes[2].legend(); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config["project_dir"], "training_history_v8.png"), dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"⚠️  plot_history: {e}")


# ============================================================
# 🚀 ANA EĞİTİM FONKSİYONU
# ============================================================
def train(config: dict):
    device = config["device"]
    print(f"\n{'='*60}")
    print(f"🚀 SigLIP2 v8 — Düzeltilmiş Versiyon")
    print(f"   Device: {device}  |  AMP: {config['use_amp']}")
    print(f"   LR: {config['learning_rate']}  |  Focal: {config['use_focal_loss']}")
    print(f"   Warmup: {config['warmup_epochs']} epoch")
    print(f"{'='*60}\n")

    # ── ADIM 1: Drive → SSD ────────────────────────────────
    copy_dataset_to_ssd(DRIVE_DATASET, LOCAL_DATASET)

    train_ds, val_ds, test_ds = build_datasets(config)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"\n📊 Toplam: {total} görsel\n")

    nw = config["num_workers"]
    kw = dict(
        batch_size         = config["batch_size"],
        num_workers        = nw,
        pin_memory         = True,
        persistent_workers = nw > 0,
        prefetch_factor    = 4 if nw > 0 else None,
    )

    if config.get("use_balanced_sampler", True):
        sampler      = make_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True,  **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)

    val_loader  = DataLoader(val_ds,  shuffle=False, drop_last=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **kw)

    # ── Model ──────────────────────────────────────────────
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
            model = torch.compile(model)
            print("⚡ torch.compile aktif")
        except Exception:
            pass

    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])

    # ── Optimizer ──────────────────────────────────────────
    bb_params  = [p for p in model.vision_model.parameters()  if p.requires_grad]
    clf_params = [p for p in model.classifier.parameters()    if p.requires_grad]

    optimizer = optim.AdamW([
        {"params": bb_params,  "lr": config["learning_rate"] * 0.01, "name": "backbone"},
        {"params": clf_params, "lr": config["learning_rate"],        "name": "classifier"},
    ], weight_decay=config["weight_decay"])

    # ── Scheduler ──────────────────────────────────────────
    warmup = config["warmup_epochs"]
    warmup_sched = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: (e + 1) / max(warmup, 1) if e < warmup else 1.0
    )
    if config.get("use_cosine_restarts"):
        cosine_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config["cosine_T0"], T_mult=config["cosine_T_mult"], eta_min=1e-7)
    else:
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config["epochs"] - warmup, 1), eta_min=1e-7)

    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], [warmup])

    # ── SWA ────────────────────────────────────────────────
    use_swa   = config.get("use_swa", True)
    swa_model = swa_sched = None
    if use_swa:
        swa_model = optim.swa_utils.AveragedModel(model)
        swa_sched = optim.swa_utils.SWALR(optimizer, swa_lr=config["swa_lr"])

    early_stop = EarlyStopping(config["early_stopping_patience"], config["early_stopping_min_delta"])
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "val_auc": [], "val_f1": [], "val_precision": [], "val_recall": [],
        "best_threshold": 0.5
    }
    best_f1  = 0.0
    best_thr = 0.5

    for epoch in range(config["epochs"]):
        apply_progressive_unfreeze(model, epoch, config, optimizer)

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, config, epoch, scaler)
        val_loss, val_acc, val_auc, val_f1, val_prec, val_rec, val_probs, val_lbl = validate(
            model, val_loader, config, epoch)

        if use_swa and epoch >= config["swa_start_epoch"]:
            swa_model.update_parameters(model)
            swa_sched.step()
            scheduler.step()
        else:
            scheduler.step()

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("val_loss", val_loss),   ("val_acc", val_acc),
                     ("val_auc", val_auc),      ("val_f1", val_f1),
                     ("val_precision", val_prec), ("val_recall", val_rec)]:
            history[k].append(v)

        gap = tr_acc - val_acc
        if gap < 5:
            tag = "🟢 İyi"
        elif gap < 10:
            tag = "🟡 Dikkat"
        else:
            tag = "🔴 OVERFIT"

        elapsed = time.time() - t0
        clf_lr  = optimizer.param_groups[1]['lr']
        bb_lr   = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{config['epochs']}  ({elapsed:.0f}s)  "
              f"LR clf={clf_lr:.2e}  bb={bb_lr:.2e}")
        print(f"  Train  Loss:{tr_loss:.4f}  Acc:{tr_acc:.2f}%")
        print(f"  Val    Loss:{val_loss:.4f}  Acc:{val_acc:.2f}%  AUC:{val_auc:.4f}  F1:{val_f1:.4f}")
        print(f"  Gap: {gap:.2f}%  {tag}")

        plot_epoch_snapshot(history, epoch, config)

        if val_f1 > best_f1 and val_auc > config["min_auc_to_save"]:
            best_f1  = val_f1
            best_thr = find_best_threshold(val_probs, val_lbl)
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

    # SWA BN güncelleme
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

    # ── TEST ────────────────────────────────────────────────
    print(f"\n{'='*60}\n🧪 TEST DEĞERLENDİRMESİ\n{'='*60}")

    if not os.path.exists(config["save_path"]):
        print("⚠️  Model kaydedilmedi.")
        return model, history, {}

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

    gap_gen = abs(best_f1 - t_f1)
    gen_tag = "🟢 İyi" if gap_gen < 0.03 else "🟡 Kabul" if gap_gen < 0.07 else "🔴 Sorun"

    print(f"\n   Test  Acc:{t_acc:.2f}%  F1:{t_f1:.4f}  AUC:{t_auc:.4f}")
    print(f"   Prec:{t_prec:.4f}  Rec:{t_rec:.4f}")
    print(f"   Val-Test F1 farkı: {gap_gen:.4f}  {gen_tag}")

    tp = int(((t_pred==1)&(t_lbl==1)).sum()); fp = int(((t_pred==1)&(t_lbl==0)).sum())
    tn = int(((t_pred==0)&(t_lbl==0)).sum()); fn = int(((t_pred==0)&(t_lbl==1)).sum())
    plot_confusion_matrix(tp, fp, tn, fn,
                          os.path.join(config["project_dir"], "confusion_matrix_test_v8.png"),
                          thr, f"TEST t={thr:.2f}")

    if config.get("use_tta", True):
        print("\n🔁 TTA...")
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
    torch.save(ckpt, config["save_path"])

    print(f"\n💾 Çıktılar → {config['project_dir']}")
    print("✅ Tamamlandı.")
    return model, history, results

# ============================================================
# ▶️  ÇALIŞTIRMA
# ============================================================
if __name__ == "__main__":
    model, history, results = train(CONFIG)

