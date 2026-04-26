import torch

print(torch.cuda.is_available())
try:
    print(torch.cuda.get_device_name(0))
except Exception:
    pass

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import torchvision.transforms as transforms
import cv2
import numpy as np
import math
import glob
import os
import sys
import platform
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy import fftpack
from contextlib import nullcontext
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import albumentations as A

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

warnings.filterwarnings('ignore')

EPSILON = 1e-8
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v3.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Colors:
    HEADER = '\033[95m';
    BLUE = '\033[94m';
    CYAN = '\033[96m'
    GREEN = '\033[92m';
    YELLOW = '\033[93m';
    RED = '\033[91m'
    ENDC = '\033[0m';
    BOLD = '\033[1m'


@dataclass
class TrainingConfig:
    train_dir: str = r"train_dir"
    val_dir: str = r"val_dir"
    test_dir: str = r"test_dir"

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
    use_optical_flow: bool = True

    device: torch.device = field(init=False)
    mixed_precision: bool = field(init=False)
    num_workers: int = field(init=False)
    gpu_name: str = field(init=False)
    gpu_memory_gb: float = field(init=False)

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = torch.cuda.is_available()
        self.gpu_name = ""
        self.gpu_memory_gb = 0.0
        self.num_workers = 2 if platform.system() == 'Linux' else 0

        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

            if 'A100' in self.gpu_name:
                self.batch_size = 32 if self.gpu_memory_gb >= 75 else 24
                self.seq_length = 8 if self.gpu_memory_gb >= 75 else 6
                self.num_workers = 4
                self.gradient_accumulation_steps = 1
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            elif 'V100' in self.gpu_name or 'T4' in self.gpu_name:
                self.batch_size = 12 if 'V100' in self.gpu_name else 8
                self.seq_length = 5
                self.num_workers = 2
                torch.backends.cudnn.benchmark = True
            else:
                self.batch_size = 8
                self.seq_length = 4
                torch.backends.cudnn.benchmark = True
        else:
            self.batch_size = 2
            self.seq_length = 4
            self.gradient_accumulation_steps = 8
            self.use_optical_flow = False

        if self.use_mixup:
            self.label_smoothing = min(self.label_smoothing, 0.02)

    def auto_scale_for_dataset(self, total_videos: int):
        is_a100 = 'A100' in self.gpu_name
        is_high_end = is_a100 or ('V100' in self.gpu_name)

        if total_videos < 5_000:
            self.epochs = 20;
            self.patience = 8;
            self.label_smoothing = 0.05
        elif total_videos < 15_000:
            self.epochs = 30;
            self.patience = 10
            if not is_a100:
                self.batch_size = min(self.batch_size, 12 if is_high_end else 8)
        elif total_videos < 50_000:
            self.epochs = 50;
            self.learning_rate = 8e-5;
            self.patience = 12
            if not is_a100:
                self.batch_size = 16 if is_high_end else (8 if torch.cuda.is_available() else 2)
        else:
            self.epochs = 80;
            self.learning_rate = 1e-4;
            self.patience = 15
            if is_a100:
                self.gradient_accumulation_steps = 1
            else:
                self.batch_size = 32 if is_high_end else (16 if torch.cuda.is_available() else 2)
                self.gradient_accumulation_steps = 2


config = TrainingConfig()


def get_frequency_features(img_rgb: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gf = gray.astype(np.float32) / 255.0

        b1 = cv2.GaussianBlur(gf, (3, 3), 0.5)
        b2 = cv2.GaussianBlur(gf, (7, 7), 1.5)
        b3 = cv2.GaussianBlur(gf, (11, 11), 3.0)
        dog = np.abs(b1 - b2) + np.abs(b2 - b3) + np.abs(gf - b1)
        ch0 = dog / (dog.max() + EPSILON)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=5))
        ch1 = (lap / (lap.max() + EPSILON)).astype(np.float32)

        r, g, b = [img_rgb[:, :, i].astype(np.float32) for i in range(3)]
        gr = cv2.Sobel(r, cv2.CV_64F, 1, 1, ksize=3)
        gg = cv2.Sobel(g, cv2.CV_64F, 1, 1, ksize=3)
        gb_ = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3)
        incon = (np.abs(gr - gg) + np.abs(gg - gb_)) / 2.0
        ch2 = (incon / (incon.max() + EPSILON)).astype(np.float32)

        return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)
    except Exception:
        return np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.float32)


def extract_dense_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_norm = mag / (mag.max() + EPSILON)
        ang_norm = ang / (2 * np.pi)
        consistency = mag_norm * np.sin(ang_norm * np.pi)
        return np.stack([mag_norm, ang_norm, consistency], axis=-1).astype(np.float32)
    except Exception:
        return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.float32)


def get_frame_indices_uniform(total: int, n: int) -> List[int]:
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n, dtype=int).tolist()


def get_frame_indices_augmented(total: int, n: int) -> List[int]:
    if total <= n:
        return list(range(total))
    n_global, n_local = n // 2, n - n // 2
    global_idx = np.linspace(0, total - 1, n_global, dtype=int)
    center = np.random.randint(total // 4, max(total // 4 + 1, 3 * total // 4))
    window = max(total // 10, n_local * 3)
    lo, hi = max(0, center - window // 2), min(total - 1, center + window // 2)
    local_idx = np.linspace(lo, hi, n_local, dtype=int)
    return sorted(set(global_idx.tolist() + local_idx.tolist()))[:n]


class NumpyTransform:
    def __init__(self, size: int, mean: tuple, std: tuple):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        if img.shape[:2] != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        normed = (img - self.mean) / (self.std + EPSILON)
        return torch.from_numpy(normed.transpose(2, 0, 1)).float()


class DualStreamVideoDataset(Dataset):
    def __init__(
            self,
            video_paths: List[str],
            labels: List[int],
            seq_length: int = 8,
            augment: bool = True,
            use_optical_flow: bool = False
    ):
        self.paths = video_paths
        self.labels = labels
        self.seq = seq_length
        self.augment = augment
        self.use_flow = use_optical_flow

        s = config.img_size
        self.rgb_tf = NumpyTransform(s, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.freq_tf = NumpyTransform(s, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.aug_pipeline = None
        if augment and HAS_ALBUMENTATIONS:
            self.aug_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
                A.ColorJitter(p=0.3),
                A.CoarseDropout(p=0.2),
            ])

    def _dummy(self, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ch = 9 if self.use_flow else 6
        return (
            torch.zeros(self.seq, ch, config.img_size, config.img_size),
            torch.tensor(label, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        label = self.labels[idx]

        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return self._dummy(label)

            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n <= 0:
                cap.release()
                return self._dummy(label)

            if n >= self.seq:
                if self.augment:
                    start = np.random.randint(0, n - self.seq + 1)
                    idxs = list(range(start, start + self.seq))
                else:
                    idxs = np.linspace(0, n - 1, self.seq, dtype=int).tolist()
            else:
                idxs = list(range(n))

            frames_selected = []
            current_idx = 0
            max_idx = max(idxs)

            while True:
                ret, frame = cap.read()
                if not ret or current_idx > max_idx:
                    break
                if current_idx in idxs:
                    frames_selected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                current_idx += 1
            cap.release()

            if len(frames_selected) == 0:
                return self._dummy(label)

            rgb_list, freq_list, flow_list = [], [], []
            prev_gray = None

            for frame_rgb in frames_selected:
                frame_resized = cv2.resize(frame_rgb, (config.img_size, config.img_size))

                if self.aug_pipeline:
                    try:
                        frame_resized = self.aug_pipeline(image=frame_resized)['image']
                    except Exception:
                        pass

                rgb_list.append(self.rgb_tf(frame_resized))
                freq_list.append(self.freq_tf(get_frequency_features(frame_resized)))

                if self.use_flow:
                    curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
                    if prev_gray is None:
                        flow = np.zeros((config.img_size, config.img_size, 3), dtype=np.float32)
                    else:
                        flow = extract_dense_optical_flow(prev_gray, curr_gray)
                    prev_gray = curr_gray
                    flow_list.append(self.freq_tf(flow))

            def _pad(lst, zero_shape):
                while len(lst) < self.seq:
                    lst.append(lst[-1].clone() if lst else torch.zeros(zero_shape))

            zero3 = (3, config.img_size, config.img_size)
            _pad(rgb_list, zero3)
            _pad(freq_list, zero3)
            if self.use_flow:
                _pad(flow_list, zero3)

            tensors = [torch.stack(rgb_list[:self.seq]), torch.stack(freq_list[:self.seq])]
            if self.use_flow:
                tensors.append(torch.stack(flow_list[:self.seq]))

            video = torch.cat(tensors, dim=1)
            return video.contiguous(), torch.tensor(label, dtype=torch.float32)

        except Exception:
            return self._dummy(label)


def get_files_from_subdirs(directory: str) -> Tuple[List[str], List[str]]:
    real_dir = os.path.join(directory, 'real')
    fake_dir = os.path.join(directory, 'fake')
    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        return [], []

    real = []
    for ext in VIDEO_EXTENSIONS:
        real.extend([str(p) for p in Path(real_dir).rglob(f'*{ext}')])

    fake = []
    for ext in VIDEO_EXTENSIONS:
        fake.extend([str(p) for p in Path(fake_dir).rglob(f'*{ext}')])

    return real, fake


def load_and_balance_data(base_dir: str) -> Tuple[List[str], List[int]]:
    real, fake = get_files_from_subdirs(base_dir)
    if not real or not fake:
        return [], []
    n = min(len(real), len(fake))
    real = np.random.choice(real, n, replace=False).tolist()
    fake = np.random.choice(fake, n, replace=False).tolist()
    return real + fake, [0] * n + [1] * n


def custom_collate_fn(batch):
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.stack(labels)


class FrequencyStreamCNN(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x)).flatten(1)


class DualStreamExtractor(nn.Module):
    def __init__(self, use_optical_flow: bool = False, use_ckpt: bool = False):
        super().__init__()
        from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
        eff = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        self.rgb_features = eff.features
        self.rgb_pool = eff.avgpool
        rgb_dim = 2048

        freq_in = 6 if use_optical_flow else 3
        self.freq_stream = FrequencyStreamCNN(in_ch=freq_in)
        freq_dim = self.freq_stream.feature_dim

        total = rgb_dim + freq_dim
        self.feature_dim = 1024

        self.rgb_gate = nn.Sequential(nn.Linear(total, 1), nn.Sigmoid())
        self.freq_gate = nn.Sequential(nn.Linear(total, 1), nn.Sigmoid())

        self.fusion = nn.Sequential(
            nn.Linear(total, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.use_flow = use_optical_flow
        self.use_ckpt = use_ckpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[:2]
        rgb_in = x[:, :, :3].reshape(B * S, 3, x.size(3), x.size(4))
        freq_in = x[:, :, 3:6]
        if self.use_flow and x.size(2) > 6:
            freq_in = torch.cat([freq_in, x[:, :, 6:9]], dim=2)
        freq_in = freq_in.reshape(B * S, -1, x.size(3), x.size(4))

        if self.use_ckpt and self.training:
            rf = grad_checkpoint(self.rgb_features, rgb_in, use_reentrant=False)
        else:
            rf = self.rgb_features(rgb_in)

        rf = torch.flatten(self.rgb_pool(rf), 1)
        ff = self.freq_stream(freq_in)

        cat = torch.cat([rf, ff], dim=1)

        rg = self.rgb_gate(cat)
        fg = self.freq_gate(cat)
        gated = torch.cat([rf * rg, ff * fg], dim=1)

        return self.fusion(gated).view(B, S, -1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int = 32, d_model: int = 1024):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d: int = 1024, heads: int = 8, layers: int = 4, dropout: float = 0.2, max_seq: int = 32):
        super().__init__()
        self.pos = LearnablePositionalEncoding(max_seq, d)
        self.diff_proj = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.diff_alpha = nn.Parameter(torch.tensor(0.1))

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = torch.zeros_like(x)
        diff[:, 1:] = x[:, 1:] - x[:, :-1]
        x = x + self.diff_alpha * self.diff_proj(diff)
        return self.norm(self.tf(self.pos(x)))


class ResBlock(nn.Module):
    def __init__(self, in_d: int, out_d: int):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU())
        self.skip = nn.Linear(in_d, out_d) if in_d != out_d else nn.Identity()
        self.norm = nn.LayerNorm(out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.main(x) + self.skip(x))


class DeepfakeDetectorV3(nn.Module):
    def __init__(self, use_optical_flow: bool = False):
        super().__init__()
        self.spatial = DualStreamExtractor(
            use_optical_flow=use_optical_flow,
            use_ckpt=config.use_gradient_checkpointing
        )
        d = self.spatial.feature_dim

        self.temporal = TemporalTransformerEncoder(
            d=d,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.transformer_dropout
        )

        self.attn_pool = nn.MultiheadAttention(d, num_heads=8, batch_first=True, dropout=0.1)

        self.classifier = nn.Sequential(
            ResBlock(d, 512), nn.Dropout(0.3),
            ResBlock(512, 256), nn.Dropout(0.2),
            ResBlock(256, 64), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(x)

    def forward_temporal(self, feats: torch.Tensor) -> torch.Tensor:
        t = self.temporal(feats)
        q = t.mean(dim=1, keepdim=True)
        att, _ = self.attn_pool(q, t, t)
        return self.classifier(att.squeeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_temporal(self.extract_spatial(x))


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha;
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        inp_f, tgt_f = inp.view(-1), tgt.view(-1)
        bce = self.bce(inp_f, tgt_f)
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class LabelSmoothBCE(nn.Module):
    def __init__(self, s: float = 0.05):
        super().__init__()
        self.s = s

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        inp_f, tgt_f = inp.view(-1), tgt.view(-1)
        t = tgt_f * (1 - self.s) + 0.5 * self.s
        return F.binary_cross_entropy_with_logits(inp_f, t)


class CombinedLoss(nn.Module):
    def __init__(self, smoothing: float = 0.05, focal_w: float = 0.3):
        super().__init__()
        self.ls = LabelSmoothBCE(smoothing)
        self.fl = FocalLoss()
        self.fw = focal_w

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return (1 - self.fw) * self.ls(inp, tgt) + self.fw * self.fl(inp, tgt)


class OnlineHardMiningLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, ratio: float = 0.7):
        super().__init__()
        self.base = base_loss
        self.ratio = ratio

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        inp_f, tgt_f = inp.view(-1), tgt.view(-1)
        with torch.no_grad():
            per = F.binary_cross_entropy_with_logits(inp_f, tgt_f, reduction='none')
            k = max(1, int(self.ratio * len(per)))
            thr = torch.topk(per, k).values[-1]
            mask = per >= thr
        if mask.sum() == 0:
            return self.base(inp, tgt)
        return self.base(inp_f[mask], tgt_f[mask])


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup: Dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup.clear()


def feature_mixup(fa, fb, la, lb, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha)
    return lam * fa + (1 - lam) * fb, lam * la + (1 - lam) * lb


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def fn(cur: int) -> float:
        if cur < warmup_steps:
            return float(cur) / max(1, warmup_steps)
        prog = (cur - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))

    return optim.lr_scheduler.LambdaLR(optimizer, fn)


def plot_training_history(history: Dict, path: str = 'training_history.png'):
    try:
        ep = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training History — V3 Dual Stream', fontsize=16, fontweight='bold')

        axes[0, 0].plot(ep, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(ep, history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss');
        axes[0, 0].legend();
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(ep, history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(ep, history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy (%)');
        axes[0, 1].legend();
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(ep, history['val_precision'], 'g-', label='Precision', linewidth=2)
        axes[1, 0].plot(ep, history['val_recall'], 'm-', label='Recall', linewidth=2)
        axes[1, 0].plot(ep, history['val_f1'], 'c-', label='F1', linewidth=2)
        if 'val_auc' in history and history['val_auc']:
            axes[1, 0].plot(ep, history['val_auc'], 'k-', label='AUC-ROC', linewidth=2)
        axes[1, 0].set_title('Val Metrics');
        axes[1, 0].legend();
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].axis('off')
        best_ep = int(np.argmax(history['val_acc'])) + 1
        txt = (f"Best Epoch:     {best_ep}\n"
               f"Best Val Acc:   {max(history['val_acc']):.2f}%\n"
               f"Best F1:        {max(history['val_f1']):.3f}\n"
               f"Best AUC-ROC:   {max(history.get('val_auc', [0])):.3f}\n"
               f"Final Val Acc:  {history['val_acc'][-1]:.2f}%")
        axes[1, 1].text(0.1, 0.5, txt, fontsize=13, family='monospace',
                        va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        pass


def plot_detailed_metrics(history: Dict, path: str = 'detailed_metrics.png'):
    try:
        ep = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Detaylı Metrikler — V3', fontsize=16, fontweight='bold')

        axes[0, 0].plot(ep, history['train_loss'], 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[0, 0].plot(ep, history['val_loss'], 'r-', label='Val', linewidth=2, alpha=0.7)
        if len(ep) >= 3:
            z = np.polyfit(list(ep), history['val_loss'], 2)
            axes[0, 0].plot(ep, np.poly1d(z)(list(ep)), 'r--', alpha=0.5, label='Trend')
        axes[0, 0].set_title('Loss + Trend');
        axes[0, 0].legend();
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(ep, history['train_acc'], 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[0, 1].plot(ep, history['val_acc'], 'r-', label='Val', linewidth=2, alpha=0.7)
        axes[0, 1].fill_between(list(ep), history['train_acc'], history['val_acc'],
                                alpha=0.2, color='gray')
        axes[0, 1].set_title('Accuracy (Gap = Overfit)');
        axes[0, 1].legend();
        axes[0, 1].grid(alpha=0.3)

        overfit = [tr - va for tr, va in zip(history['train_acc'], history['val_acc'])]
        colors = ['green' if x < 5 else 'orange' if x < 10 else 'red' for x in overfit]
        axes[0, 2].bar(list(ep), overfit, color=colors, alpha=0.6)
        axes[0, 2].axhline(y=5, color='orange', linestyle='--', label='Uyarı (5%)')
        axes[0, 2].axhline(y=10, color='red', linestyle='--', label='Overfit (10%)')
        axes[0, 2].set_title('Overfitting Göstergesi');
        axes[0, 2].legend();
        axes[0, 2].grid(alpha=0.3)

        axes[1, 0].plot(history['val_recall'], history['val_precision'], 'bo-', linewidth=2, markersize=6)
        for i, e in enumerate(ep):
            if i % max(1, len(list(ep)) // 8) == 0:
                axes[1, 0].annotate(f'E{e}', (history['val_recall'][i], history['val_precision'][i]), fontsize=7,
                                    alpha=0.7)
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].set_xlabel('Recall');
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(ep, history['val_f1'], 'g-', linewidth=2.5, marker='o', markersize=4)
        axes[1, 1].fill_between(list(ep), history['val_f1'], alpha=0.3, color='green')
        bi = int(np.argmax(history['val_f1']))
        axes[1, 1].axvline(x=bi + 1, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].scatter([bi + 1], [history['val_f1'][bi]], color='red', s=200, zorder=5, marker='*')
        axes[1, 1].annotate(f"Best: {history['val_f1'][bi]:.3f}",
                            xy=(bi + 1, history['val_f1'][bi]), xytext=(10, 10),
                            textcoords='offset points', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        if 'val_auc' in history and history['val_auc']:
            ax2 = axes[1, 1].twinx()
            ax2.plot(ep, history['val_auc'], 'k--', linewidth=1.5, alpha=0.6, label='AUC-ROC')
            ax2.set_ylabel('AUC-ROC');
            ax2.legend(loc='lower right')
        axes[1, 1].set_title('F1 Progression');
        axes[1, 1].grid(alpha=0.3)

        axes[1, 2].axis('off')
        best_ep_idx = int(np.argmax(history['val_acc']))
        improvement = history['val_acc'][-1] - history['val_acc'][0]
        final_overfit = history['train_acc'][-1] - history['val_acc'][-1]
        rows = [
            ['Metric', 'Value'],
            ['─' * 18, '─' * 12],
            ['Best Epoch', str(best_ep_idx + 1)],
            ['Best Val Acc', f"{max(history['val_acc']):.2f}%"],
            ['Best F1', f"{max(history['val_f1']):.3f}"],
            ['Best AUC-ROC', f"{max(history.get('val_auc', [0])):.3f}"],
            ['Final Val Acc', f"{history['val_acc'][-1]:.2f}%"],
            ['Improvement', f"{improvement:+.2f}%"],
            ['Final Overfit', f"{final_overfit:.2f}%"],
            ['Total Epochs', str(len(list(ep)))],
        ]
        tbl = axes[1, 2].table(cellText=rows, loc='center', cellLoc='left', colWidths=[0.6, 0.4])
        tbl.auto_set_font_size(False);
        tbl.set_fontsize(11);
        tbl.scale(1, 2)
        for i in range(2):
            tbl[(0, i)].set_facecolor('#4CAF50')
            tbl[(0, i)].set_text_props(weight='bold', color='white')
        axes[1, 2].set_title('Eğitim İstatistikleri', fontweight='bold', fontsize=12, pad=20)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        pass


def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, path: str = 'confusion_matrix.png'):
    try:
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
               title='Confusion Matrix', ylabel='Gerçek', xlabel='Tahmin')
        thr = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thr else 'black',
                    fontsize=20, fontweight='bold')
        acc = (tp + tn) / (tp + tn + fp + fn + EPSILON) * 100
        prec = tp / (tp + fp + EPSILON)
        rec = tp / (tp + fn + EPSILON)
        f1 = 2 * prec * rec / (prec + rec + EPSILON)
        ax.text(0.02, 0.98,
                f'Acc:{acc:.1f}%  P:{prec:.3f}  R:{rec:.3f}  F1:{f1:.3f}',
                transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        pass


def evaluate_dataset(model: nn.Module, loader, base_loss: nn.Module, device, scaler, label: str = "Test") -> Dict:
    model.eval()
    total_loss = 0.0
    correct = total = 0
    tp = fp = tn = fn = 0
    all_probs: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for videos, labels_batch in tqdm(loader, desc=f"  {label}", leave=False):
            videos, labels_batch = (videos.to(device, non_blocking=True),
                                    labels_batch.to(device, non_blocking=True))
            amp_ctx = torch.amp.autocast('cuda') if scaler else nullcontext()
            with amp_ctx:
                out = model(videos)
                loss = base_loss(out, labels_batch)

            total_loss += loss.item()
            probs = torch.sigmoid(out.view(-1))
            pred = (probs > 0.6).float()
            labs = labels_batch.view(-1)
            total += labs.size(0)
            correct += (pred == labs).sum().item()
            tp += ((pred == 1) & (labs == 1)).sum().item()
            fp += ((pred == 1) & (labs == 0)).sum().item()
            tn += ((pred == 0) & (labs == 0)).sum().item()
            fn += ((pred == 0) & (labs == 1)).sum().item()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    acc = 100 * correct / (total + EPSILON)
    prec = tp / (tp + fp + EPSILON)
    rec = tp / (tp + fn + EPSILON)
    f1 = 2 * prec * rec / (prec + rec + EPSILON)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return {
        'loss': total_loss / max(1, len(loader)),
        'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'probs': all_probs, 'labels': all_labels
    }


def train():
    train_p, train_l = load_and_balance_data(config.train_dir)
    val_p, val_l = load_and_balance_data(config.val_dir)

    if not train_p or not val_p:
        return

    config.auto_scale_for_dataset(len(train_p))

    train_ds = DualStreamVideoDataset(train_p, train_l, config.seq_length,
                                      augment=True, use_optical_flow=config.use_optical_flow)
    val_ds = DualStreamVideoDataset(val_p, val_l, config.seq_length,
                                    augment=False, use_optical_flow=config.use_optical_flow)

    use_pin = torch.cuda.is_available()
    use_pers = config.num_workers > 0
    pf = 4 if config.num_workers > 0 else None

    train_dl = DataLoader(train_ds, config.batch_size, shuffle=True,
                          num_workers=config.num_workers, pin_memory=use_pin,
                          prefetch_factor=pf, persistent_workers=use_pers,
                          collate_fn=custom_collate_fn, drop_last=True)
    val_dl = DataLoader(val_ds, config.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=use_pin,
                        prefetch_factor=pf, persistent_workers=use_pers,
                        collate_fn=custom_collate_fn, drop_last=False)

    model = DeepfakeDetectorV3(use_optical_flow=config.use_optical_flow).to(config.device)

    bb_ids = (set(id(p) for p in model.spatial.rgb_features.parameters()) |
              set(id(p) for p in model.spatial.rgb_pool.parameters()))
    optimizer = optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) in bb_ids],
         'lr': config.learning_rate * config.backbone_lr_mult},
        {'params': [p for p in model.parameters() if id(p) not in bb_ids],
         'lr': config.learning_rate}
    ], weight_decay=config.weight_decay, betas=(0.9, 0.999))

    steps_per_epoch = max(1, len(train_dl) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    base_loss = CombinedLoss(smoothing=config.label_smoothing)
    criterion = (OnlineHardMiningLoss(base_loss, config.hard_mining_ratio)
                 if config.use_hard_mining else base_loss)

    scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
    ema = EMA(model, config.ema_decay) if config.use_ema else None

    best_acc, patience_ctr = 0.0, 0
    history: Dict[str, List] = {k: [] for k in [
        'train_loss', 'train_acc', 'val_loss', 'val_acc',
        'val_precision', 'val_recall', 'val_f1', 'val_auc'
    ]}

    for epoch in range(config.epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_dl), total=len(train_dl),
                    desc=f"{Colors.GREEN}Ep {epoch + 1}/{config.epochs} [TRAIN]{Colors.ENDC}",
                    leave=True, file=sys.stdout)

        for bi, (vids, labs) in pbar:
            vids = vids.to(config.device, non_blocking=True)
            labs = labs.to(config.device, non_blocking=True)

            amp_ctx = torch.amp.autocast('cuda') if scaler else nullcontext()
            with amp_ctx:
                spatial = model.extract_spatial(vids)
                actual_labs = labs

                if (config.use_mixup and vids.size(0) >= 2 and np.random.random() < 0.5):
                    idx = torch.randperm(vids.size(0), device=vids.device)
                    spatial, actual_labs = feature_mixup(
                        spatial, spatial[idx], labs, labs[idx], config.mixup_alpha)

                out = model.forward_temporal(spatial)
                loss = criterion(out, actual_labs)
                loss = loss / config.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (bi + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if scaler:
                    scaler.step(optimizer);
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if ema:
                    ema.update(model)

            t_loss += loss.item() * config.gradient_accumulation_steps
            pred = (torch.sigmoid(out.view(-1)) > 0.5).float()
            t_total += labs.size(0)
            t_correct += (pred == labs.view(-1)).sum().item()
            pbar.set_postfix_str(f"loss={t_loss / (bi + 1):.4f} acc={100 * t_correct / t_total:.1f}%")

        train_loss = t_loss / len(train_dl)
        train_acc = 100 * t_correct / t_total

        if ema:
            ema.apply(model)
        val_pbar = tqdm(val_dl,
                        desc=f"{Colors.BLUE}Ep {epoch + 1}/{config.epochs} [ VAL ]{Colors.ENDC}",
                        leave=True, file=sys.stdout)

        model.eval()
        v_loss = v_correct = v_total = 0
        tp = fp = tn = fn = 0
        all_probs: List[float] = []
        all_labels_: List[int] = []

        with torch.no_grad():
            for bi_v, (vids, labs) in enumerate(val_pbar):
                vids = vids.to(config.device, non_blocking=True)
                labs = labs.to(config.device, non_blocking=True)
                amp_ctx = torch.amp.autocast('cuda') if scaler else nullcontext()
                with amp_ctx:
                    out = model(vids)
                    loss = base_loss(out, labs)

                v_loss += loss.item()
                probs = torch.sigmoid(out.view(-1))
                pred = (probs > 0.6).float()
                labs_f = labs.view(-1)
                v_total += labs.size(0)
                v_correct += (pred == labs_f).sum().item()
                tp += ((pred == 1) & (labs_f == 1)).sum().item()
                fp += ((pred == 1) & (labs_f == 0)).sum().item()
                tn += ((pred == 0) & (labs_f == 0)).sum().item()
                fn += ((pred == 0) & (labs_f == 1)).sum().item()
                all_probs.extend(probs.cpu().numpy())
                all_labels_.extend(labs_f.cpu().numpy())
                val_pbar.set_postfix_str(f"loss={v_loss / (bi_v + 1):.4f} acc={100 * v_correct / v_total:.1f}%")

        if ema:
            ema.restore(model)

        val_loss = v_loss / len(val_dl)
        val_acc = 100 * v_correct / v_total
        prec = tp / (tp + fp + EPSILON)
        rec = tp / (tp + fn + EPSILON)
        f1 = 2 * prec * rec / (prec + rec + EPSILON)
        try:
            auc = roc_auc_score(all_labels_, all_probs)
        except Exception:
            auc = 0.0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(prec)
        history['val_recall'].append(rec)
        history['val_f1'].append(f1)
        history['val_auc'].append(auc)

        try:
            plot_training_history(history, 'training_history.png')
            plot_confusion_matrix(tp, fp, tn, fn, 'confusion_matrix_latest.png')
            plot_detailed_metrics(history, 'detailed_metrics.png')
        except Exception:
            pass

        if val_acc > best_acc:
            best_acc, patience_ctr = val_acc, 0
            if ema:
                ema.apply(model)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f'best_model_v3_ep{epoch + 1}_acc{val_acc:.2f}_{timestamp}.pth'
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc, 'val_loss': val_loss,
                'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
                'config': config,
                'history': history,
                'timestamp': timestamp,
            }
            torch.save(ckpt, fname)
            torch.save(ckpt, 'best_model_v3.pth')
            if ema:
                ema.restore(model)
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                break

    test_p, test_l = load_and_balance_data(config.test_dir)
    if test_p:
        ckpt = torch.load('best_model_v3.pth', map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        test_ds = DualStreamVideoDataset(test_p, test_l, config.seq_length,
                                         augment=False, use_optical_flow=config.use_optical_flow)
        test_dl = DataLoader(test_ds, config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=use_pin,
                             collate_fn=custom_collate_fn)
        if ema:
            ema.apply(model)
        res = evaluate_dataset(model, test_dl, base_loss, config.device, scaler, label="Test")
        if ema:
            ema.restore(model)
        plot_confusion_matrix(res['tp'], res['fp'], res['tn'], res['fn'], 'confusion_matrix_test.png')

    try:
        plot_training_history(history, 'training_history_final.png')
        plot_detailed_metrics(history, 'detailed_metrics_final.png')
    except Exception:
        pass


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise