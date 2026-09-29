from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Features are computed on a thumbnail so cost does not grow with resolution
# and every region of the image contributes (not just the first rows).
ANALYSIS_SIZE = 256

@dataclass
class ImageFeatures:
    width: int
    height: int
    contrast: float
    edge_density: float
    exposure_penalty: float
    artifact_risk: float
    is_blank: bool

def _thumbnail_array(image):
    import numpy as np
    thumb = image.copy()
    thumb.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    return np.asarray(thumb, dtype=np.float32)

def compute_image_features(path: str | Path) -> Tuple[object, ImageFeatures]:
    import numpy as np
    from PIL import Image
    image = Image.open(path).convert("RGB")
    if image.width == 0 or image.height == 0:
        return image, ImageFeatures(0, 0, 0, 0, 1, 1, True)
    rgb = _thumbnail_array(image)
    gray = rgb.mean(axis=2)
    std = float(gray.std())
    contrast = min(1.0, std / 128)
    # Penalise only clipped exposure: dark or bright scenes are fine unless
    # most pixels sit at the extremes.
    clipped = float(((gray < 8) | (gray > 247)).mean())
    exposure = min(1.0, max(0.0, (clipped - 0.25) / 0.5))
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        gx = np.abs(np.diff(gray, axis=1))[:-1, :]
        gy = np.abs(np.diff(gray, axis=0))[:, :-1]
        edge_density = float(((gx + gy) > 24).mean())
    else:
        edge_density = 0.0
    blank = std < 4.5 and edge_density < 0.005
    artifact = min(1.0, (1.0 - contrast) * 0.5 + (0.7 if blank else 0.0))
    return image, ImageFeatures(image.width, image.height, contrast, edge_density, exposure, artifact, blank)

def color_coverage(image, rgb: Tuple[int, int, int], tolerance: int = 85) -> float:
    import numpy as np
    pixels = _thumbnail_array(image.convert("RGB")).reshape(-1, 3)
    if pixels.size == 0:
        return 0.0
    distance = np.sqrt(((pixels - np.asarray(rgb, dtype=np.float32)) ** 2).sum(axis=1))
    return float((distance <= tolerance).mean())
