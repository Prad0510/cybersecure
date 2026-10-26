import os
import io
import cv2
import numpy as np
from PIL import Image
from typing import Union, List

# Try importing PyTorch & Torchvision for deep vision backbone
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision.models import MobileNet_V3_Small_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VisualEncoder:
    """
    Deep Convolutional Vision Feature Encoder:
    Replaces handcrafted heuristics with a pretrained MobileNetV3-Small Deep Neural Network backbone.
    
    Architecture & Representations:
    - Backbone: MobileNetV3-Small (penultimate global pooled feature vector, 576-d)
    - Image normalization: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    - Spatial canvas: 224x224 RGB tensor
    - Projection: Linear projection layer (576-d -> 128-d)
    - Geometric normalization: L2 unit sphere normalization
    - Fallback: Multi-scale spatial/edge/DCT histogram extractor if Torch is unavailable.
    """

    def __init__(self, visual_dim: int = 128, device: str = "cpu"):
        self.visual_dim = visual_dim
        self.device = device
        self.deep_model = None
        self.projector = None
        self.transform = None
        self.backend = "heuristic_fallback"

        if TORCH_AVAILABLE:
            self._init_deep_backbone()

    def _init_deep_backbone(self):
        """Initialize MobileNetV3-Small deep vision feature extractor."""
        try:
            self.device_obj = torch.device(self.device)
            # Load pretrained MobileNetV3-Small (efficient, edge-ready, ~9MB)
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base_model = models.mobilenet_v3_small(weights=weights)
            
            # Extract features before the final classifier logits
            # MobileNetV3 has .features -> .avgpool -> .classifier
            # Stripping the final classification layers leaves global pooled 576-d features
            base_model.classifier = nn.Identity()
            base_model.eval()
            base_model.to(self.device_obj)
            self.deep_model = base_model

            # Calibrated projection layer (576 -> visual_dim)
            # Seeded orthogonal projection for stable dense representations
            torch.manual_seed(42)
            proj = nn.Linear(576, self.visual_dim, bias=False)
            nn.init.orthogonal_(proj.weight)
            proj.eval()
            proj.to(self.device_obj)
            self.projector = proj

            # Preprocessing pipeline
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.backend = "mobilenet_v3_small_deep_cnn"
        except Exception as exc:
            self.deep_model = None
            self.backend = "heuristic_fallback"

    def _load_as_pil(self, image_input: Union[str, bytes, Image.Image, np.ndarray]) -> Image.Image:
        """Convert any supported image input format to a 3-channel RGB PIL Image."""
        if isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                return None
            return Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                # Handle OpenCV BGR numpy array
                rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_arr)
        return None

    def _encode_deep(self, pil_img: Image.Image) -> np.ndarray:
        """Pass image through MobileNetV3 deep CNN backbone."""
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device_obj)
        with torch.no_grad():
            raw_features = self.deep_model(tensor)          # (1, 576)
            projected = self.projector(raw_features)       # (1, 128)
            normed = torch.nn.functional.normalize(projected, p=2, dim=1)
        return normed.squeeze(0).cpu().numpy().astype(np.float32)

    def _encode_fallback(self, img_bgr: np.ndarray) -> np.ndarray:
        """Fallback classical spatial/edge/DCT visual feature extractor."""
        canvas = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        h, w = canvas.shape[:2]
        features = []

        # 64d Quadrant Color & Intensity
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        half_h, half_w = h // 2, w // 2
        quadrants = [
            canvas[0:half_h, 0:half_w],
            canvas[0:half_h, half_w:w],
            canvas[half_h:h, 0:half_w],
            canvas[half_h:h, half_w:w]
        ]
        for quad in quadrants:
            quad_gray = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)
            hist, _ = np.histogram(quad_gray, bins=16, range=(0, 256), density=True)
            features.extend(hist)

        # 32d Structural Edge Spectrum
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_hist, _ = np.histogram(grad_mag, bins=16, range=(0, 255), density=True)
        features.extend(edge_hist)

        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        lap_hist, _ = np.histogram(laplacian, bins=16, range=(0, 255), density=True)
        features.extend(lap_hist)

        # 32d Frequency / DCT Texture Coefficients
        float_gray = np.float32(gray) / 255.0
        dct = cv2.dct(float_gray)
        dct_coeffs = dct[:8, :4].flatten()
        features.extend(dct_coeffs)

        feature_vec = np.array(features, dtype=np.float32)
        if len(feature_vec) > self.visual_dim:
            feature_vec = feature_vec[:self.visual_dim]
        elif len(feature_vec) < self.visual_dim:
            pad = np.zeros(self.visual_dim - len(feature_vec), dtype=np.float32)
            feature_vec = np.concatenate([feature_vec, pad])

        norm = np.linalg.norm(feature_vec)
        if norm > 1e-9:
            feature_vec = feature_vec / norm
        return feature_vec.astype(np.float32)

    def encode_image(self, image_input: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode an image into a 128-dimensional dense visual embedding.
        Accepts: filepath (str), raw image bytes, PIL Image, or numpy array.
        """
        if image_input is None:
            return np.zeros(self.visual_dim, dtype=np.float32)

        try:
            pil_img = self._load_as_pil(image_input)
            if pil_img is None:
                return np.zeros(self.visual_dim, dtype=np.float32)

            if self.deep_model is not None:
                return self._encode_deep(pil_img)
            else:
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                return self._encode_fallback(img_bgr)
        except Exception:
            return np.zeros(self.visual_dim, dtype=np.float32)

    def encode_batch(self, images: List[Union[str, bytes, Image.Image, np.ndarray]]) -> np.ndarray:
        """Batch encode multiple images into a (N, 128) feature matrix."""
        return np.vstack([self.encode_image(img) for img in images])
