import gradio as gr
import time
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchaudio
import torchaudio.transforms as T
import cv2
import numpy as np
from pathlib import Path
import warnings
import yaml
from PIL import Image as pil_image

EFFORT_FOLDER = 'image_detector'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), EFFORT_FOLDER))

warnings.filterwarnings('ignore')

# ==================== AI VIDEO DETECTOR KODLARI ====================
EPSILON = 1e-8


class TrainingConfig:

    def __init__(self):
        self.use_optical_flow = True
        self.batch_size = 8
        self.seq_length = 10
        self.img_size = 224


def get_advanced_frequency_features(img_rgb: np.ndarray) -> np.ndarray:
    try:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        kernel_size = 7
        img_float = img_gray.astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 1.5)
        high_freq = img_float - blur

        high_freq_enhanced = np.abs(high_freq)
        dct_approx = (high_freq_enhanced - high_freq_enhanced.min()) / (
                high_freq_enhanced.max() - high_freq_enhanced.min() + EPSILON
        )

        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        edge_norm = (edge_magnitude - edge_magnitude.min()) / (
                edge_magnitude.max() - edge_magnitude.min() + EPSILON
        )

        color_variance = np.var(img_rgb.astype(np.float32), axis=2) / 255.0
        color_norm = (color_variance - color_variance.min()) / (
                color_variance.max() - color_variance.min() + EPSILON
        )

        freq_features = np.stack([dct_approx, edge_norm, color_norm], axis=-1)
        return freq_features.astype(np.float32)

    except Exception:
        return np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.float32)


def extract_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    try:
        corners = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01,
            minDistance=10, blockSize=7
        )

        if corners is not None and len(corners) > 10:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, corners, None,
                winSize=(15, 15), maxLevel=2,
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

    except Exception:
        return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.float32)


def get_frame_indices(frame_count: int, sequence_length: int) -> list:
    if frame_count > sequence_length:
        return np.linspace(0, frame_count - 1, sequence_length, dtype=int).tolist()
    return list(range(frame_count))


class EfficientNetExtractor(nn.Module):

    def __init__(self, use_optical_flow: bool = True):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, *x.size()[2:])
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(batch_size, seq_len, -1)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, feature_dim: int, num_heads: int = 8,
                 num_layers: int = 4, dropout: float = 0.2):
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
        self.spatial_extractor = EfficientNetExtractor(use_optical_flow=use_optical_flow)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_extractor(x)
        temporal_features = self.temporal_encoder(spatial_features)
        query = temporal_features.mean(dim=1, keepdim=True)
        attended, _ = self.attention_pool(query, temporal_features, temporal_features)
        final_features = attended.squeeze(1)
        output = self.classifier(final_features)
        return output


class VideoDetectorModel:

    def __init__(self, model_path: str = "video_model.pth"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_info = {}
        self.sequence_length = 10
        self.img_size = 224
        self.is_loaded = False

        if Path(model_path).exists():
            try:
                self.load_model()
                self.is_loaded = True
            except Exception as e:
                print(f"Video model yüklenemedi: {e}")
                self.is_loaded = False
        else:
            print(f"Video model dosyası bulunamadı: {model_path}")
            self.is_loaded = False

    def load_model(self):
        """Video modelini yükleme."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        config = checkpoint.get('config', None)
        use_optical_flow = True

        if config and hasattr(config, 'use_optical_flow'):
            use_optical_flow = config.use_optical_flow

        self.model = AdvancedDeepfakeDetector(use_optical_flow=use_optical_flow)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.model_info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_acc': checkpoint.get('val_acc', 'N/A'),
            'f1': checkpoint.get('f1', 'N/A'),
            'use_optical_flow': use_optical_flow
        }

        print(f" Video AI Detector yüklendi!")

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            cap.release()
            return None

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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

                freq_features = get_advanced_frequency_features(frame_rgb)
                frame_tensor = transform(freq_features)
                frames.append(frame_tensor)

                if self.model_info['use_optical_flow'] and prev_gray is not None:
                    flow_features = extract_optical_flow(prev_gray, frame_gray)
                    flow_tensor = transform(flow_features)
                    optical_flows.append(flow_tensor)

                prev_gray = frame_gray.copy()

                if len(frames) >= self.sequence_length:
                    break

        except Exception as e:
            cap.release()
            return None
        finally:
            cap.release()

        while len(frames) < self.sequence_length:
            if frames:
                frames.append(frames[-1].clone())
                if self.model_info['use_optical_flow'] and optical_flows:
                    optical_flows.append(optical_flows[-1].clone())
            else:
                frames.append(torch.zeros((3, self.img_size, self.img_size)))
                if self.model_info['use_optical_flow']:
                    optical_flows.append(torch.zeros((3, self.img_size, self.img_size)))

        video_tensor = torch.stack(frames[:self.sequence_length])

        if self.model_info['use_optical_flow']:
            while len(optical_flows) < self.sequence_length - 1:
                if optical_flows:
                    optical_flows.append(optical_flows[-1].clone())
                else:
                    optical_flows.append(torch.zeros((3, self.img_size, self.img_size)))

            flow_tensor = torch.stack(optical_flows[:self.sequence_length - 1])
            dummy_flow = torch.zeros_like(frames[0])
            flow_tensor = torch.cat([dummy_flow.unsqueeze(0), flow_tensor], dim=0)
            video_tensor = torch.cat([video_tensor, flow_tensor], dim=1)

        video_tensor = video_tensor.unsqueeze(0)
        return video_tensor

    def predict(self, video_path: str):
        """Video üzerinde tahmin yapma."""
        if not self.is_loaded:
            return 0.5, "Video AI Detector modeli yüklenemedi!"

        video_tensor = self.process_video(video_path)
        if video_tensor is None:
            return 0.5, "Video işlenemedi!"

        self.model.eval()
        video_tensor = video_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(video_tensor)
            probability = torch.sigmoid(output).item()

        is_fake = probability > 0.5
        confidence = probability if is_fake else (1 - probability)

        if is_fake:
            msg = f"🚨 AI-Generated  - Güven: {confidence * 100:.1f}%"
        else:
            msg = f"✅ Real (Gerçek) - Güven: {confidence * 100:.1f}%"

        return probability, msg


# ==================== AUDIO DETECTOR ====================

class AudioDeepfakeModel:
    """LFCC tabanlı ses tespit etme """

    def __init__(self, model_path: str = "audio.pth"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_loaded = False
        self.input_dim = 12840

        self.lfcc_transform = T.LFCC(sample_rate=16000, n_lfcc=40).to(self.device)

        if Path(model_path).exists():
            try:
                self.load_model()
                self.is_loaded = True
            except Exception as e:
                print(f"Audio model yüklenemedi: {e}")
                self.is_loaded = False
        else:
            print(f"Audio model dosyası bulunamadı: {model_path}")
            self.is_loaded = False

    def load_model(self):
        """Ses modelini yükleme."""
        try:
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2)
            )

            checkpoint = torch.load(self.model_path, map_location=self.device)


            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f" Checkpoint'ten model_state_dict yüklendi")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint


            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()

            print(f" AI Audio Detector yüklendi!")

        except Exception as e:
            print(f" Audio model yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            raise
    def preprocess_audio(self, audio_path: str):
        """Ses dosyası işleme."""
        try:
            from scipy.io import wavfile

            ext = os.path.splitext(audio_path)[1].lower()

            if ext not in ['.wav']:
                return None

            sr, waveform_np = wavfile.read(audio_path)

            if waveform_np.dtype == np.int16:
                waveform_np = waveform_np.astype(np.float32) / 32768.0
            elif waveform_np.dtype == np.int32:
                waveform_np = waveform_np.astype(np.float32) / 2147483648.0
            else:
                waveform_np = waveform_np.astype(np.float32)

            if len(waveform_np.shape) > 1:
                waveform_np = np.mean(waveform_np, axis=1)

            waveform = torch.from_numpy(waveform_np).unsqueeze(0)

            if sr != 16000:
                resampler = T.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            target_length = 64000
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))

            return waveform.to(self.device)

        except Exception as e:
            print(f" Audio preprocessing hatası: {e}")
            return None
    def predict(self, audio_path: str):
        """Ses dosyası üzerinde tahmni yapma."""
        if not self.is_loaded:
            return 0.5, "AI Audio Detector modeli yüklenemedi!"

        waveform = self.preprocess_audio(audio_path)
        if waveform is None:
            return 0.5, "Ses dosyası işlenemedi!"

        try:
            self.model.eval()
            with torch.no_grad():
                feats = self.lfcc_transform(waveform).squeeze(1)
                feats = feats.reshape(feats.size(0), -1)

                outputs = self.model(feats)
                probs = torch.softmax(outputs, dim=1)

                spoof_prob = probs[0][0].item()
                bonafide_prob = probs[0][1].item()

                ai_probability = spoof_prob

                is_fake = spoof_prob > bonafide_prob
                confidence = max(spoof_prob, bonafide_prob)

                if is_fake:
                    msg = f"🚨 AI-Generated (Spoof/Deepfake) - Güven: {confidence * 100:.1f}%"
                else:
                    msg = f"✅ Real (Bonafide/Gerçek) - Güven: {confidence * 100:.1f}%"

                return ai_probability, msg

        except Exception as e:
            print(f"Audio prediction hatası: {e}")
            return 0.5, f"Tahmin sırasında hata oluştu: {str(e)}"


# ==================== IMAGE DEEPFAKE DETECTOR ====================

class ImageDeepfakeModel:
    """Görsel tahmini"""

    def __init__(self, detector_config: str = "image_detector/training/config/detector/effort.yaml",
                 weights: str = "image_model.pth"):
        self.detector_config = detector_config
        self.weights = weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.is_loaded = False

        if Path(weights).exists():
            try:
                self.load_model()
                self.is_loaded = True
            except Exception as e:
                print(f"Image model yüklenemedi: {e}")
                print(f"Detay: {str(e)}")
                self.is_loaded = False
        else:
            print(f" Image model dosyası bulunamadı: {weights}")
            self.is_loaded = False

    def load_model(self):
        try:

            if Path(self.detector_config).exists():
                with open(self.detector_config, 'r') as f:
                    self.config = yaml.safe_load(f)
                print(f" Config yüklendi: {self.detector_config}")
            else:
                print(f"️ Config dosyası bulunamadı: {self.detector_config}")
                self.config = {
                    'model_name': 'effort',
                    'backbone_name': 'clip_RN50'
                }

            try:
                from detectors import DETECTOR
                model_class = DETECTOR[self.config['model_name']]
                self.model = model_class(self.config).to(self.device)
                print(f" Model mimarisi yüklendi")
            except ImportError as e:
                print(f" DETECTOR modülü bulunamadı: {e}")
                self.is_loaded = False
                return

            ckpt = torch.load(self.weights, map_location=self.device, weights_only=False)
            state = ckpt.get("state_dict", ckpt)

            state = {k.replace("module.", ""): v for k, v in state.items()}


            self.model.load_state_dict(state, strict=False)
            self.model.eval()

            self.is_loaded = True
            print(f" AI Image Detector yüklendi!")

        except Exception as e:
            print(f" Image model yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False

    def preprocess_image(self, img_path: str):
        """Görsel dosyalarını işleme """
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                return None

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])

            img_tensor = transform(pil_image.fromarray(img_rgb)).unsqueeze(0)
            return img_tensor.to(self.device)

        except Exception as e:
            print(f"Image preprocessing hatası: {e}")
            return None

    def predict(self, img_path: str):
        """Görsel üzerinde tahmni yapma."""
        if not self.is_loaded:
            return 0.5, " AI Image Detector modeli yüklenemedi! ."

        img_tensor = self.preprocess_image(img_path)
        if img_tensor is None:
            return 0.5, " Görsel işlenemedi!"

        try:
            self.model.eval()
            with torch.no_grad():
                data_dict = {
                    'image': img_tensor,
                    'label': torch.tensor([0]).to(self.device)
                }

                predictions = self.model(data_dict, inference=True)

                prob = predictions['prob'].squeeze().cpu().numpy()
                cls = predictions['cls'].squeeze().cpu().numpy()

                if isinstance(prob, np.ndarray):
                    fake_prob = float(prob.flatten()[0])
                else:
                    fake_prob = float(prob)

                if isinstance(cls, np.ndarray):
                    cls_value = int(cls.flatten()[0])
                else:
                    cls_value = int(cls)

                fake_prob = float(prob.flatten()[0]) if isinstance(prob, np.ndarray) else float(prob)

                is_fake = fake_prob >= 0.5
                confidence = fake_prob if is_fake else (1 - fake_prob)

                if is_fake:
                    msg = f"🚨 AI-Generated (FAKE) - Güven: {confidence * 100:.1f}%"
                else:
                    msg = f"✅ Real (GERÇEK) - Güven: {confidence * 100:.1f}%"

                return fake_prob, msg


        except Exception as e:
            print(f" Image prediction hatası: {e}")
            import traceback
            traceback.print_exc()
            return 0.5, f"Tahmin sırasında hata oluştu: {str(e)}"

# ==================== MODEL YÜKLEME ====================

def load_all_models():
    """Tüm modelleri yükle"""
    try:
        print("=" * 70)
        print("🚀 AI BUSTER - Modeller Yükleniyor...")
        print("=" * 70)

        # Image Deepfake Detector (EFFORT - GenImage)
        print("\n📸 Image Model yükleniyor...")
        image_model = ImageDeepfakeModel(
            detector_config="image_detector/training/config/detector/effort.yaml",
            weights="image_model.pth"
        )

        # Video AI Detector
        print("\n🎬 Video Model yükleniyor...")
        video_model = VideoDetectorModel("video_model.pth")

        # Audio Deepfake Detector
        print("\n🎵 Audio Model yükleniyor...")
        audio_model = AudioDeepfakeModel("audio_model.pth")

        print("\n" + "=" * 70)
        print(" Model yükleme tamamlandı!")
        print("=" * 70 + "\n")
        return image_model, video_model, audio_model
    except Exception as e:
        print(f" Modeller yüklenirken hata çıktı: {e}")
        return None, None, None


IMAGE_MODEL, VIDEO_MODEL, AUDIO_MODEL = load_all_models()


# --- ANALİZ FONKSİYONLARI ---

def analyze_image(file_path):
    """Görsel analizi """
    if IMAGE_MODEL is None or not isinstance(IMAGE_MODEL, ImageDeepfakeModel):
        return 0.5, "Görsel Modeli Yüklü Değil!"



    score, msg = IMAGE_MODEL.predict(file_path)
    return score, msg


def analyze_video(file_path):
    """Video analizi """
    if VIDEO_MODEL is None or not isinstance(VIDEO_MODEL, VideoDetectorModel):
        return 0.5, "Video Modeli Yüklü Değil!"

    if not VIDEO_MODEL.is_loaded:
        return 0.5, "Video AI Detector modeli yüklenemedi!"

    score, msg = VIDEO_MODEL.predict(file_path)
    return score, msg


def analyze_audio(file_path):
    """Ses analizi """
    if AUDIO_MODEL is None or not isinstance(AUDIO_MODEL, AudioDeepfakeModel):
        return 0.5, "Ses Modeli Yüklü Değil!"

    if not AUDIO_MODEL.is_loaded:
        return 0.5, "Audio Deepfake Detector modeli yüklenemedi!"

    # Dosya formatı kontrolü
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.wav']:
        return 0.5, f" Sadece WAV formatı desteklenmektedir. Lütfen ses dosyanızı WAV'a çevirin. (Mevcut: {ext})"

    score, msg = AUDIO_MODEL.predict(file_path)
    return score, msg


# --- ANA YÖNLENDİRİCİ ---

def main_process(file):
    if file is None:
        return None, "Lütfen bir dosya yükleyin!"

    ext = os.path.splitext(file.name)[1].lower()

    try:
        if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            score, msg = analyze_image(file.name)
        elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
            score, msg = analyze_video(file.name)
        elif ext in [".mp3", ".wav", ".m4a", ".flac"]:
            score, msg = analyze_audio(file.name)
        else:
            return None, "Desteklenmeyen dosya formatı!"

        return {"Yapay Zeka (AI)": score, "İnsan Yapımı (Real)": 1 - score}, msg

    except Exception as e:
        return None, f"İşlem sırasında bir hata oluştu: {str(e)}"


# --- ARAYÜZ ---

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@900&family=Montserrat:wght@400;700&display=swap');

.big-title {
    font-family: 'Kanit', sans-serif !important;
    font-size: 85px !important;
    font-weight: 900 !important;
    line-height: 0.9 !important;
    margin: 0 !important;
    color: #1e272e !important;
    letter-spacing: -2px;
}

.sub-title {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 22px !important;
    font-weight: 400 !important;
    opacity: 0.8;
    margin-top: -5px !important;
}
"""

logo_path = r"logo.jpeg"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row(elem_classes="side-by-side"):
        with gr.Column(scale=1, min_width=150):
            if os.path.exists(logo_path):
                gr.Image(logo_path, show_label=False, container=False,
                         width=150, interactive=False)
            else:
                gr.Markdown("### LOGO YOK!")

        with gr.Column(scale=4):
            gr.Markdown(
                f"""
                <div class="big-title">AI BUSTER</div>
                <div style="font-size: 24px; font-weight: bold; opacity: 0.7;">
                    Görsel, Video ve Ses Analiz Platformu
                </div>
                """
            )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Dosyayı Sürükleyin veya Seçin",
                file_types=["image", "video", "audio"]
            )
            run_btn = gr.Button("🔍 ANALİZİ BAŞLAT", variant="primary")

        with gr.Column(scale=1):
            result_label = gr.Label(label="Yapay Zeka Olasılığı")
            result_text = gr.Textbox(label="Sistem Mesajı", interactive=False)

    run_btn.click(
        fn=main_process,
        inputs=file_input,
        outputs=[result_label, result_text],
        show_progress=True
    )

    gr.Markdown("---")
    gr.Markdown("""
    ### 📁 Kabul Edilen Dosya Türleri:
    - **Görsel:** .jpg, .png, .webp, .bmp (AI Image Detection! ✅)
    - **Video:** .mp4, .mov, .avi, .mkv (AI Video Detection! ✅)
    - **Ses:** .wav (AI Audio Detection! ✅)
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)