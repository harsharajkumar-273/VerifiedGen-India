from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass
class ImageFeatures:
    width: int
    height: int
    contrast: float
    edge_density: float
    exposure_penalty: float
    artifact_risk: float
    is_blank: bool

def _pixels(path: str | Path):
    from PIL import Image
    image = Image.open(path).convert("RGB")
    return image, list(image.getdata())

def compute_image_features(path: str | Path) -> Tuple[object, ImageFeatures]:
    image, pixels = _pixels(path)
    if not pixels:
        return image, ImageFeatures(0, 0, 0, 0, 1, 1, True)
    values = [sum(pixel) / 3 for pixel in pixels]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    contrast = min(1.0, variance ** 0.5 / 128)
    exposure = min(1.0, abs(mean - 128) / 128)
    unique = len({tuple(pixel) for pixel in pixels[: min(len(pixels), 5000)]})
    edge_density = min(1.0, unique / 500.0)
    blank = variance < 20 and unique < 10
    artifact = min(1.0, (1.0 - contrast) * 0.5 + (1.0 if blank else 0.0) * 0.7)
    return image, ImageFeatures(image.width, image.height, contrast, edge_density, exposure, artifact, blank)

def color_coverage(image, rgb: Tuple[int, int, int], tolerance: int = 85) -> float:
    pixels = list(image.convert("RGB").getdata())
    if not pixels:
        return 0.0
    hits = sum(1 for pixel in pixels if sum((pixel[i] - rgb[i]) ** 2 for i in range(3)) ** 0.5 <= tolerance)
    return hits / len(pixels)

