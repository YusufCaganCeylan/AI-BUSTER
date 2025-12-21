import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import cv2
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from scipy import fftpack
import albumentations as A
import sys
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')


EPSILON = 1e-8  # Sayısal kararlılık için sabit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class TrainingConfig:
    """Eğitim için konfigürasyon"""
    batch_size: int = 8  # GPU ve Veri setine göre otomatik ayarlanmaktadır.
    epochs: int = 5
    learning_rate: float = 1e-4
    seq_length: int = 4  # Veri setine göre otomatik ayarlanmaktadır.
    img_size: int = 224
    patience: int = 7
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    use_gradient_checkpointing: bool = False
    use_optical_flow: bool = True
    fast_optical_flow: bool = True

    def __post_init__(self):
        #GPU'ya göre otomatik konfigürasyon
        import platform

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.gpu_name = None
        self.gpu_memory_gb = 0
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU Detected: {self.gpu_name}")
            logger.info(f"GPU Memory: {self.gpu_memory_gb:.1f} GB")


        if platform.system() == 'Windows' or 'COLAB_GPU' in os.environ:
            self.num_workers = 2


        self.mixed_precision = torch.cuda.is_available()


        if torch.cuda.is_available() and 'A100' in self.gpu_name:
            logger.info(f"\n{'='*70}")
            logger.info(f" A100 GPU DETECTED - ENABLING HIGH-PERFORMANCE MODE")
            logger.info(f"{'='*70}")

            if self.gpu_memory_gb >= 75:
                self.batch_size = 32
                self.seq_length = 8
                logger.info("   80GB A100 - Using MAXIMUM settings")
            else:
                self.batch_size = 24
                self.seq_length = 6
                logger.info("   40GB A100 - Using HIGH-PERFORMANCE settings")

            self.num_workers = 4
            self.use_gradient_checkpointing = True
            self.mixed_precision = True
            self.gradient_accumulation_steps = 1
            

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            logger.info(f"   Batch Size: {self.batch_size}")
            logger.info(f"   Sequence Length: {self.seq_length}")
            logger.info(f"   Num Workers: {self.num_workers}")
            logger.info(f"   Gradient Checkpointing: {self.use_gradient_checkpointing}")
            logger.info(f"   TF32 Enabled: True (8x speedup on matrix ops)")
            logger.info(f"   cuDNN Benchmark: True")
            logger.info(f"{'='*70}\n")


        elif torch.cuda.is_available() and ('V100' in self.gpu_name or 'T4' in self.gpu_name):
            logger.info(f"\n {self.gpu_name} detected - Optimizing...")
            self.batch_size = 12 if 'V100' in self.gpu_name else 8
            self.seq_length = 5
            self.num_workers = 2
            self.use_gradient_checkpointing = True
            torch.backends.cudnn.benchmark = True
            logger.info(f"  Batch Size: {self.batch_size}, Seq Length: {self.seq_length}\n")


        elif torch.cuda.is_available():
            logger.info(f"\n⚡ {self.gpu_name} detected - Using standard GPU settings")
            self.batch_size = 8
            self.seq_length = 4
            self.num_workers = 2
            torch.backends.cudnn.benchmark = True


        if not torch.cuda.is_available():
            self.batch_size = 4
            self.seq_length = 3
            self.num_workers = 2
            self.use_gradient_checkpointing = True
            self.use_optical_flow = False

        logger.info(f"Platform: {platform.system()}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.mixed_precision}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Seq Length: {self.seq_length}")
        logger.info(f"Num Workers: {self.num_workers}")
        logger.info(f"Optical Flow: {self.use_optical_flow} (Fast: {self.fast_optical_flow})")

    def auto_scale_for_dataset(self, total_videos: int):
        """
        GPU ve Verisetine göre hiperparametleri otomatik ayarlama
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"AUTO-SCALING based on dataset size: {total_videos} videos")
        logger.info(f"{'='*70}")

        is_a100 = self.gpu_name and 'A100' in self.gpu_name
        is_high_end_gpu = is_a100 or (self.gpu_name and 'V100' in self.gpu_name)

        if total_videos < 5000:
            logger.info("SMALL DATASET MODE (<5K videos)")
            self.epochs = 5  # Prevent overfitting
            self.label_smoothing = 0.15  # More regularization
            self.patience = 5

            if not is_a100:
                self.seq_length = 4 if self.seq_length > 3 else self.seq_length
                self.batch_size = min(self.batch_size, 8)
            
            logger.info("  - Reduced epochs (5) to prevent overfitting")
            logger.info("  - Increased label smoothing (0.15)")

        elif total_videos < 15000:
            logger.info("MEDIUM DATASET MODE (5K-15K videos)")
            self.epochs = 30
            self.label_smoothing = 0.1
            self.patience = 7
            
            if not is_a100:
                self.seq_length = 5 if self.seq_length > 3 else self.seq_length
                self.batch_size = min(self.batch_size, 12 if is_high_end_gpu else 8)
            
            logger.info("  - Standard epochs (30)")

        elif total_videos < 50000:
            logger.info("LARGE DATASET MODE (15K-50K videos)")
            self.epochs = 50
            self.learning_rate = 2e-4
            self.label_smoothing = 0.1
            self.patience = 10
            
            if not is_a100:
                self.seq_length = 6 if self.seq_length > 3 else self.seq_length
                self.batch_size = 16 if is_high_end_gpu else (8 if torch.cuda.is_available() else 4)
            
            logger.info("  - Extended epochs (50) for better convergence")
            logger.info("  - Higher learning rate (2e-4)")

        else:
            logger.info("XLARGE DATASET MODE (50K+ videos)")
            self.epochs = 100
            self.learning_rate = 3e-4
            self.label_smoothing = 0.05
            self.patience = 15
            
            if is_a100:
                self.gradient_accumulation_steps = 1
            else:
                self.seq_length = 8 if self.seq_length > 3 else self.seq_length
                self.batch_size = 32 if is_high_end_gpu else (16 if torch.cuda.is_available() else 4)
                self.gradient_accumulation_steps = 2
            
            logger.info("  - Maximum epochs (100)")
            logger.info("  - Gradient accumulation" if not is_a100 else "  - Large native batch size (A100)")

        logger.info(f"{'='*70}")
        logger.info(f"FINAL SETTINGS:")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch Size: {self.batch_size}")
        logger.info(f"  Seq Length: {self.seq_length}")
        logger.info(f"  Learning Rate: {self.learning_rate}")
        logger.info(f"  Label Smoothing: {self.label_smoothing}")
        logger.info(f"  Patience: {self.patience}")
        logger.info(f"{'='*70}\n")



config = TrainingConfig()

# Desteklenen video formatları
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']


def extract_dct_features(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Sıkıştırma kaynaklı bozulmaları tespit etmek için Discrete Cosine Transform (DCT) özelliklerini çıkartma

    Bu fonksiyon, JPEG standardına uygun olarak 8x8 bloklar üzerinde DCT hesaplar.

    """
    try:
        frame_ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
        y_channel = frame_ycrcb[:, :, 0].astype(np.float32)

        # 8x8 bloklarda işleme (JPEG standartları)
        h, w = y_channel.shape
        h_blocks = h // 8
        w_blocks = w // 8

        if h_blocks == 0 or w_blocks == 0:
            return np.zeros((h, w), dtype=np.float32)

        dct_image = np.zeros((h_blocks * 8, w_blocks * 8), dtype=np.float32)

        for i in range(h_blocks):
            for j in range(w_blocks):
                block = y_channel[i*8:(i+1)*8, j*8:(j+1)*8]
                dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                dct_image[i*8:(i+1)*8, j*8:(j+1)*8] = dct_block

        # [0, 1]'e normalize etme
        dct_norm = np.abs(dct_image)
        dct_min, dct_max = dct_norm.min(), dct_norm.max()
        if dct_max > dct_min:
            dct_norm = (dct_norm - dct_min) / (dct_max - dct_min)

        return dct_norm
    except Exception as e:
        logger.warning(f"DCT extraction failed: {e}")
        return np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.float32)


def get_advanced_frequency_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Frekans özelliklerini çıkartma

    """
    try:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        kernel_size = 7
        img_float = img_gray.astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 1.5)
        high_freq = img_float - blur

        high_freq_enhanced = np.abs(high_freq)
        dct_approx = (high_freq_enhanced - high_freq_enhanced.min()) / (high_freq_enhanced.max() - high_freq_enhanced.min() + EPSILON)


        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_norm = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + EPSILON)


        color_variance = np.var(img_rgb.astype(np.float32), axis=2) / 255.0
        color_norm = (color_variance - color_variance.min()) / (color_variance.max() - color_variance.min() + EPSILON)


        freq_features = np.stack([
            dct_approx,
            edge_norm,
            color_norm
        ], axis=-1)

        return freq_features.astype(np.float32)

    except Exception as e:
        logger.warning(f"Frequency feature extraction failed: {e}")
        return np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.float32)


def extract_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
   Hareket özelliklerini çıkartmak için Optical Flow kullanma

    """
    try:

        corners = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=200, d
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        if corners is not None and len(corners) > 10:

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, corners, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )


            h, w = prev_gray.shape
            flow_map = np.zeros((h, w, 2), dtype=np.float32)


            good_old = corners[status == 1]
            good_new = next_points[status == 1]

            for old, new in zip(good_old, good_new):
                x_old, y_old = old.ravel()
                x_new, y_new = new.ravel()


                x_int, y_int = int(x_old), int(y_old)
                if 0 <= x_int < w and 0 <= y_int < h:

                    dx, dy = x_new - x_old, y_new - y_old


                    for dy_offset in range(-2, 3):
                        for dx_offset in range(-2, 3):
                            ny, nx = y_int + dy_offset, x_int + dx_offset
                            if 0 <= ny < h and 0 <= nx < w:
                                flow_map[ny, nx] = [dx, dy]


            flow_map = cv2.GaussianBlur(flow_map, (5, 5), 1.0)


            mag, ang = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
            mag_norm = mag / (mag.max() + EPSILON)
            ang_norm = ang / (2 * np.pi)
            consistency = mag_norm * np.sin(ang_norm * np.pi)

            flow_features = np.stack([mag_norm, ang_norm, consistency], axis=-1)
            return flow_features.astype(np.float32)
        else:

            return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.float32)

    except Exception as e:
        logger.warning(f"Optical flow extraction failed: {e}")
        return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.float32)


def get_frame_indices(frame_count: int, sequence_length: int) -> List[int]:
    """
    Uniform sampling

    """
    if frame_count > sequence_length:
        return np.linspace(0, frame_count - 1, sequence_length, dtype=int).tolist()
    else:
        return list(range(frame_count))


class AdvancedVideoDataset(Dataset):

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        sequence_length: int = 10,
        augment: bool = False,
        use_optical_flow: bool = True
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.augment = augment
        self.use_optical_flow = use_optical_flow

        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
            ])
        else:
            self.aug = None

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(path)


        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)


        if frame_count <= 0 or fps <= 0:
            cap.release()
            logger.warning(f"Invalid video: {path}")
            return self._get_dummy_data(label)

        frames = []
        optical_flows = []
        prev_gray = None


        indices = get_frame_indices(frame_count, self.sequence_length)

        try:
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply augmentation
                if self.aug:
                    try:
                        augmented = self.aug(image=frame_rgb)
                        frame_rgb = augmented['image']
                    except Exception as e:
                        logger.debug(f"Augmentation failed: {e}")


                freq_features = get_advanced_frequency_features(frame_rgb)


                frame_tensor = self.transform(freq_features)
                frames.append(frame_tensor)


                if self.use_optical_flow and prev_gray is not None:
                    flow_features = extract_optical_flow(prev_gray, frame_gray)
                    flow_tensor = self.transform(flow_features)
                    optical_flows.append(flow_tensor)

                prev_gray = frame_gray.copy()

                if len(frames) >= self.sequence_length:
                    break

        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
        finally:
            cap.release()


        while len(frames) < self.sequence_length:
            if frames:
                frames.append(frames[-1].clone())
                if self.use_optical_flow and optical_flows:
                    optical_flows.append(optical_flows[-1].clone())
            else:
                frames.append(torch.zeros((3, config.img_size, config.img_size)))
                if self.use_optical_flow:
                    optical_flows.append(torch.zeros((3, config.img_size, config.img_size)))

        video_tensor = torch.stack(frames[:self.sequence_length])


        if self.use_optical_flow:

            while len(optical_flows) < self.sequence_length - 1:
                if optical_flows:
                    optical_flows.append(optical_flows[-1].clone())
                else:
                    optical_flows.append(torch.zeros((3, config.img_size, config.img_size)))

            flow_tensor = torch.stack(optical_flows[:self.sequence_length-1])

            dummy_flow = torch.zeros_like(frames[0])
            flow_tensor = torch.cat([dummy_flow.unsqueeze(0), flow_tensor], dim=0)

            video_tensor = torch.cat([video_tensor, flow_tensor], dim=1)

        return video_tensor, torch.tensor(label, dtype=torch.float32)

    def _get_dummy_data(self, label: int) -> Tuple[torch.Tensor, torch.Tensor]:

        channels = 6 if self.use_optical_flow else 3
        dummy_tensor = torch.zeros((self.sequence_length, channels, config.img_size, config.img_size))
        return dummy_tensor, torch.tensor(label, dtype=torch.float32)


class EfficientNetExtractor(nn.Module):
    """
    EfficientNet-B3 tabanlı uzamsal özellik çıkarıcı.
    """

    def __init__(self, use_optical_flow: bool = True, use_checkpointing: bool = False):
        super().__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        if use_optical_flow:
            old_conv = efficientnet.features[0][0]
            new_conv = nn.Conv2d(
                6, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )


            with torch.no_grad():

                new_conv.weight[:, :3, :, :] = old_conv.weight

                new_conv.weight[:, 3:, :, :] = old_conv.weight * 0.5

            efficientnet.features[0][0] = new_conv

        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.feature_dim = 1536
        self.use_checkpointing = use_checkpointing and config.use_gradient_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, *x.size()[2:])

        # Use gradient checkpointing to save memory
        if self.use_checkpointing and self.training:
            x = checkpoint(self.features, x, use_reentrant=False)
        else:
            x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(batch_size, seq_len, -1)

        return x


class TransformerEncoder(nn.Module):
    """
    Video sekans modellemesi için transformatör tabanlı zamansal kodlayıcı
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.transformer(x)
        return self.norm(x)


class AdvancedDeepfakeDetector(nn.Module):


    def __init__(self, use_optical_flow: bool = True):
        super().__init__()

        self.use_optical_flow = use_optical_flow


        self.spatial_extractor = EfficientNetExtractor(
            use_optical_flow=use_optical_flow,
            use_checkpointing=config.use_gradient_checkpointing
        )
        feature_dim = self.spatial_extractor.feature_dim


        self.temporal_encoder = TransformerEncoder(
            feature_dim=feature_dim,
            num_heads=8,
            num_layers=4,
            dropout=0.2
        )


        self.attention_pool = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True, dropout=0.1
        )


        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )


        self._init_weights()

    def _init_weights(self):

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        spatial_features = self.spatial_extractor(x)


        temporal_features = self.temporal_encoder(spatial_features)


        query = temporal_features.mean(dim=1, keepdim=True)
        attended, _ = self.attention_pool(query, temporal_features, temporal_features)


        final_features = attended.squeeze(1)


        output = self.classifier(final_features)

        return output


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        bce_loss = self.bce_with_logits(inputs.squeeze(), targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class LabelSmoothingBCEWithLogitsLoss(nn.Module):


    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(inputs.squeeze(), targets_smooth)


def get_video_files(directory: str) -> List[str]:


    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")

    if not os.access(directory, os.R_OK):
        raise ValueError(f"Directory is not accessible: {directory}")

    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        video_files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    return video_files


def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Ensures all tensors are contiguous for better memory performance.
    """
    videos = []
    labels = []

    for video, label in batch:
        videos.append(video.contiguous())
        labels.append(label)

    videos = torch.stack(videos, dim=0)
    labels = torch.stack(labels, dim=0)

    return videos, labels


def plot_training_history(history: Dict[str, List[float]], save_path: str = 'training_plots.png'):

    try:
        epochs = range(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training History - AI Video Detector', fontsize=16, fontweight='bold')


        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)


        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)


        axes[1, 0].plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
        axes[1, 0].plot(epochs, history['val_recall'], 'm-', label='Recall', linewidth=2)
        axes[1, 0].plot(epochs, history['val_f1'], 'c-', label='F1-Score', linewidth=2)
        axes[1, 0].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)


        axes[1, 1].axis('off')
        best_epoch = np.argmax(history['val_acc']) + 1
        best_acc = max(history['val_acc'])
        best_f1 = history['val_f1'][best_epoch - 1]
        final_acc = history['val_acc'][-1]
        final_loss = history['val_loss'][-1]

        summary_text = f"""
        TRAINING SUMMARY
        {'='*40}

        Best Validation Accuracy: {best_acc:.2f}%
        Best Epoch: {best_epoch}
        Best F1-Score: {best_f1:.3f}

        Final Validation Accuracy: {final_acc:.2f}%
        Final Validation Loss: {final_loss:.4f}

        Total Epochs Trained: {len(epochs)}

        Improvement: {final_acc - history['val_acc'][0]:.2f}%
        """

        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round',
                       facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Training plots saved to: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to create training plots: {e}")
        return None


def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int,
                         save_path: str = 'confusion_matrix.png'):

    try:
        cm = np.array([[tn, fp], [fn, tp]])

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Real', 'Fake'],
               yticklabels=['Real', 'Fake'],
               title='Confusion Matrix - Final Epoch',
               ylabel='True label',
               xlabel='Predicted label')


        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=20, fontweight='bold')

        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        stats_text = f'Accuracy: {accuracy:.2f}%\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved to: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to create confusion matrix: {e}")
        return None


def plot_detailed_metrics(history: Dict[str, List[float]], save_path: str = 'detailed_metrics.png'):

    try:
        epochs = range(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Detailed Training Metrics - AI Video Detector', fontsize=16, fontweight='bold')


        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2, alpha=0.7)

        z = np.polyfit(list(epochs), history['val_loss'], 2)
        p = np.poly1d(z)
        axes[0, 0].plot(epochs, p(list(epochs)), 'r--', alpha=0.5, label='Trend')
        axes[0, 0].set_title('Loss Curves with Trend')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)


        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2, alpha=0.7)

        axes[0, 1].fill_between(list(epochs), history['train_acc'], history['val_acc'], alpha=0.2, color='gray')
        axes[0, 1].set_title('Accuracy Curves (Gap = Overfitting)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)


        overfitting = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
        color = ['green' if x < 5 else 'orange' if x < 10 else 'red' for x in overfitting]
        axes[0, 2].bar(list(epochs), overfitting, color=color, alpha=0.6)
        axes[0, 2].axhline(y=5, color='orange', linestyle='--', label='Warning (5%)')
        axes[0, 2].axhline(y=10, color='red', linestyle='--', label='Overfitting (10%)')
        axes[0, 2].set_title('Overfitting Indicator')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Train - Val Acc (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(history['val_recall'], history['val_precision'], 'bo-', linewidth=2, markersize=8)
        for i, epoch in enumerate(epochs):
            if i % max(1, len(list(epochs))//10) == 0:  # Label ~10 points
                axes[1, 0].annotate(f'E{epoch}', (history['val_recall'][i], history['val_precision'][i]),
                                   fontsize=8, alpha=0.7)
        axes[1, 0].set_title('Precision-Recall Trade-off')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([max(0, min(history['val_recall'])-0.1), min(1.05, max(history['val_recall'])+0.1)])
        axes[1, 0].set_ylim([max(0, min(history['val_precision'])-0.1), min(1.05, max(history['val_precision'])+0.1)])


        axes[1, 1].plot(epochs, history['val_f1'], 'g-', linewidth=2.5, marker='o', markersize=4)
        axes[1, 1].fill_between(list(epochs), history['val_f1'], alpha=0.3, color='green')
        best_f1_idx = np.argmax(history['val_f1'])
        axes[1, 1].axvline(x=best_f1_idx+1, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].scatter([best_f1_idx+1], [history['val_f1'][best_f1_idx]],
                          color='red', s=200, zorder=5, marker='*')
        axes[1, 1].annotate(f'Best: {history["val_f1"][best_f1_idx]:.3f}',
                          xy=(best_f1_idx+1, history['val_f1'][best_f1_idx]),
                          xytext=(10, 10), textcoords='offset points',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        axes[1, 1].set_title('F1-Score Progression')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].grid(True, alpha=0.3)


        axes[1, 2].axis('off')

        best_epoch = np.argmax(history['val_acc']) + 1
        improvement = history['val_acc'][-1] - history['val_acc'][0]
        avg_train_acc = np.mean(history['train_acc'])
        avg_val_acc = np.mean(history['val_acc'])
        final_overfit = history['train_acc'][-1] - history['val_acc'][-1]

        stats_data = [
            ['Metric', 'Value'],
            ['─' * 20, '─' * 15],
            ['Best Epoch', f'{best_epoch}'],
            ['Best Val Acc', f'{max(history["val_acc"]):.2f}%'],
            ['Best F1-Score', f'{max(history["val_f1"]):.3f}'],
            ['Final Val Acc', f'{history["val_acc"][-1]:.2f}%'],
            ['Improvement', f'{improvement:+.2f}%'],
            ['Avg Train Acc', f'{avg_train_acc:.2f}%'],
            ['Avg Val Acc', f'{avg_val_acc:.2f}%'],
            ['Final Overfit', f'{final_overfit:.2f}%'],
            ['Total Epochs', f'{len(epochs)}'],
        ]

        table = axes[1, 2].table(cellText=stats_data, loc='center', cellLoc='left',
                                colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)


        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        axes[1, 2].set_title('Training Statistics', fontweight='bold', fontsize=12, pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Detailed metrics saved to: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to create detailed metrics: {e}")
        return None


def train():

    logger.info(f" Device: {config.device}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f" Supported formats: {', '.join(VIDEO_EXTENSIONS)}\n")


    # Veri seti yoluna göre değişin
    real_videos = get_video_files("data/real")
    fake_videos = get_video_files("data/fake")

    if not real_videos or not fake_videos:
        logger.error(" ERROR: No videos found!")
        logger.error(f"Real videos: {len(real_videos)}")
        logger.error(f"AI-generated videos: {len(fake_videos)}")
        logger.error("\n Expected folder structure:")
        logger.error("   data/real/     → Real videos")
        logger.error("   data/fake/     → AI-generated videos")
        return

    logger.info(f" Dataset Statistics:")
    logger.info(f" Real videos: {len(real_videos)}")
    logger.info(f" AI-generated videos: {len(fake_videos)}")


    logger.info("\n Format Distribution:")
    for ext in VIDEO_EXTENSIONS:
        real_count = sum(1 for v in real_videos if v.lower().endswith(ext))
        fake_count = sum(1 for v in fake_videos if v.lower().endswith(ext))
        if real_count + fake_count > 0:
            logger.info(f"  {ext}: {real_count} real, {fake_count} fake")


    min_count = min(len(real_videos), len(fake_videos))
    real_videos = np.random.choice(real_videos, min_count, replace=False).tolist()
    fake_videos = np.random.choice(fake_videos, min_count, replace=False).tolist()

    video_paths = real_videos + fake_videos
    labels = [0] * len(real_videos) + [1] * len(fake_videos)


    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"\n Train set: {len(train_paths)} videos")
    logger.info(f" Validation set: {len(val_paths)} videos")


    config.auto_scale_for_dataset(len(train_paths))


    train_dataset = AdvancedVideoDataset(
        train_paths, train_labels,
        sequence_length=config.seq_length,
        augment=True,
        use_optical_flow=config.use_optical_flow
    )
    val_dataset = AdvancedVideoDataset(
        val_paths, val_labels,
        sequence_length=config.seq_length,
        augment=False,
        use_optical_flow=config.use_optical_flow
    )


    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = config.num_workers > 0 and torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=4 if config.num_workers > 0 else None,
        persistent_workers=use_persistent_workers,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
        collate_fn=custom_collate_fn
    )


    model = AdvancedDeepfakeDetector(use_optical_flow=config.use_optical_flow).to(config.device)


    if config.label_smoothing > 0:
        criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=config.label_smoothing)
    else:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)


    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )


    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )


    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None


    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0


    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    logger.info("\n Training started...\n")

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad()

        desc = f"{Colors.CYAN}Epoch {epoch+1}/{config.epochs} [TRAIN]{Colors.ENDC}"
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=desc,
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout
        )

        for batch_idx, (videos, labels_batch) in pbar:
            videos = videos.to(config.device, non_blocking=True)
            labels_batch = labels_batch.to(config.device, non_blocking=True)


            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels_batch)
                    loss = loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()


                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(videos)
                loss = criterion(outputs, labels_batch)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()


                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * config.gradient_accumulation_steps
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()
            current_acc = 100 * train_correct / train_total
            avg_loss = train_loss / (batch_idx + 1)

            postfix_str = (
                f"Loss: {Colors.YELLOW}{avg_loss:.4f}{Colors.ENDC} | "
                f"Acc: {Colors.GREEN}{current_acc:.2f}%{Colors.ENDC}"
            )
            pbar.set_postfix_str(postfix_str)
            sys.stdout.flush()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_tn = val_fn = 0

        desc = f"{Colors.BLUE}Epoch {epoch+1}/{config.epochs} [VAL]{Colors.ENDC}"
        pbar_val = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=desc,
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout
        )

        with torch.no_grad():
            for batch_idx, (videos, labels_batch) in pbar_val:
                videos = videos.to(config.device, non_blocking=True)
                labels_batch = labels_batch.to(config.device, non_blocking=True)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(videos)
                        loss = criterion(outputs, labels_batch)
                else:
                    outputs = model(videos)
                    loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()

                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()

                # Confusion matrix
                val_tp += ((predicted == 1) & (labels_batch == 1)).sum().item()
                val_fp += ((predicted == 1) & (labels_batch == 0)).sum().item()
                val_tn += ((predicted == 0) & (labels_batch == 0)).sum().item()
                val_fn += ((predicted == 0) & (labels_batch == 1)).sum().item()

                # Update progress bar
                current_val_acc = 100 * val_correct / val_total
                avg_val_loss = val_loss / (batch_idx + 1)

                postfix_str = (
                    f"Loss: {Colors.YELLOW}{avg_val_loss:.4f}{Colors.ENDC} | "
                    f"Acc: {Colors.GREEN}{current_val_acc:.2f}%{Colors.ENDC}"
                )
                pbar_val.set_postfix_str(postfix_str)
                sys.stdout.flush()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total


        precision = val_tp / (val_tp + val_fp + EPSILON)
        recall = val_tp / (val_tp + val_fn + EPSILON)
        f1 = 2 * (precision * recall) / (precision + recall + EPSILON)


        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD} Epoch {epoch+1}/{config.epochs} Summary{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

        print(f"{Colors.CYAN}Training:{Colors.ENDC}")
        print(f"  Loss: {Colors.YELLOW}{train_loss:.4f}{Colors.ENDC} | "
              f"Accuracy: {Colors.GREEN}{train_acc:.2f}%{Colors.ENDC}")

        print(f"\n{Colors.BLUE}Validation:{Colors.ENDC}")
        print(f"  Loss: {Colors.YELLOW}{val_loss:.4f}{Colors.ENDC} | "
              f"Accuracy: {Colors.GREEN}{val_acc:.2f}%{Colors.ENDC}")

        print(f"\n{Colors.BOLD}Metrics:{Colors.ENDC}")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
        print(f"  TP: {val_tp} | FP: {val_fp} | TN: {val_tn} | FN: {val_fn}")

        print(f"\n{Colors.BOLD}Learning Rate:{Colors.ENDC} {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

        sys.stdout.flush()


        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)


        try:
            plot_training_history(history, 'training_history.png')
            plot_confusion_matrix(val_tp, val_fp, val_tn, val_fn, 'confusion_matrix_latest.png')
            plot_detailed_metrics(history, 'detailed_metrics.png')
            logger.info(f"Grafikler güncellendi (3 grafik)")
        except Exception as e:
            logger.debug(f"Plot generation failed: {e}")


        scheduler.step()


        if val_acc > best_val_acc:
            best_val_acc = val_acc

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f'best_model_epoch{epoch+1}_acc{val_acc:.2f}_{timestamp}.pth'

            model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'config': config,
                'history': history,
                'timestamp': timestamp,
            }


            torch.save(model_checkpoint, model_filename)


            torch.save(model_checkpoint, 'best_model_advanced.pth')

            logger.info(f"{Colors.GREEN} Best model saved! "
                       f"(Val Acc: {val_acc:.2f}%, F1: {f1:.3f}){Colors.ENDC}")
            logger.info(f"   {model_filename}\n")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"\n{Colors.YELLOW} Early stopping triggered at epoch {epoch+1}{Colors.ENDC}")
            break

    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}Training completed!{Colors.ENDC}")
    print(f"{Colors.BOLD} Best Validation Accuracy: {best_val_acc:.2f}%{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

    logger.info("Training completed successfully!")


    logger.info("\n Generating final visualizations...")
    try:

        plot_path = plot_training_history(history, 'training_history_final.png')
        if plot_path:
            print(f"\n{Colors.GREEN} Final training plots: {plot_path}{Colors.ENDC}")


        detailed_path = plot_detailed_metrics(history, 'detailed_metrics_final.png')
        if detailed_path:
            print(f"{Colors.GREEN} Detailed metrics: {detailed_path}{Colors.ENDC}")


        cm_path = plot_confusion_matrix(val_tp, val_fp, val_tn, val_fn, 'confusion_matrix_final.png')
        if cm_path:
            print(f"{Colors.GREEN} Final confusion matrix: {cm_path}{Colors.ENDC}")

        print(f"\n{Colors.CYAN} Çıktı Dosyaları:{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Modeller:{Colors.ENDC}")
        print(f"   - best_model_advanced.pth (en iyi model)")
        print(f"   - best_model_epoch*_acc*_*.pth (versioned models)")

        print(f"\n{Colors.BOLD}Grafikler (Her epoch güncellenir):{Colors.ENDC}")
        print(f"   - training_history.png (loss & accuracy)")
        print(f"   - confusion_matrix_latest.png (confusion matrix)")
        print(f"   - detailed_metrics.png (detaylı analiz)")

        print(f"\n{Colors.BOLD}Final Grafikler:{Colors.ENDC}")
        print(f"   - training_history_final.png (final)")
        print(f"   - detailed_metrics_final.png (detaylı analiz - final)")
        print(f"   - confusion_matrix_final.png (confusion matrix - final)")

        print(f"\n{Colors.BOLD}Loglar:{Colors.ENDC}")
        print(f"   - training.log (eğitim logları)")

        print(f"\n{Colors.YELLOW} İpucu:{Colors.ENDC}")
        print(f"   - detailed_metrics.png → Overfitting analizi, PR curve, istatistikler")
        print(f"   - Her epoch sonrası grafikler otomatik güncellenir")

    except Exception as e:
        logger.error(f"Failed to generate final plots: {e}")

    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\n Training failed: {e}", exc_info=True)
        raise