"""Contrast optimization and adaptive histogram equalization."""

from typing import Tuple
import cv2
import numpy as np

from src.utils.validators import validate_image_array


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Standard global histogram equalization frequently washes out text and over-amplifies
    background noise in bright regions. CLAHE divides the image into contextual tiles
    (e.g., 8x8), equalizes each tile independently, and limits contrast amplification
    by clipping histogram bins at `clip_limit`.
    
    For 3-channel color images, CLAHE is applied exclusively to the Lightness (L) channel
    in the CIELAB color space, preserving chromatic integrity without color shift.
    
    Args:
        image: Source image array (Grayscale or BGR).
        clip_limit: Contrast clipping threshold >= 1.0 (default 2.0).
        tile_grid_size: Grid dimensions for contextual blocks (default (8, 8)).
        
    Returns:
        Contrast-equalized image array with identical channels.
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If clip_limit is < 1.0 or tile dimensions are invalid.
    """
    validate_image_array(image)
    if clip_limit < 1.0:
        raise ValueError(f"clip_limit must be >= 1.0, got {clip_limit}.")

    gw, gh = int(tile_grid_size[0]), int(tile_grid_size[1])
    if gw <= 0 or gh <= 0:
        raise ValueError(f"tile_grid_size dimensions must be positive, got {tile_grid_size}.")

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(gw, gh))

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        # Grayscale
        gray = image if image.ndim == 2 else image[:, :, 0]
        return clahe.apply(gray)

    # 3-channel Color: Convert BGR to LAB, equalize Lightness channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_equalized = clahe.apply(l_chan)
    lab_merged = cv2.merge((l_equalized, a_chan, b_chan))

    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0
) -> np.ndarray:
    """Linearly scale image brightness and contrast.
    
    output(x, y) = clip(contrast * input(x, y) + brightness, 0, 255)
    
    Args:
        image: Source image (Grayscale or Color).
        brightness: Additive brightness offset (e.g. -100 to +100, default 0).
        contrast: Multiplicative contrast scale factor >= 0.0 (e.g. 0.5 to 2.5, default 1.0).
        
    Returns:
        Adjusted image array (uint8).
    """
    validate_image_array(image)
    if contrast < 0.0:
        raise ValueError(f"Contrast factor cannot be negative, got {contrast}.")

    adjusted = cv2.convertScaleAbs(image, alpha=float(contrast), beta=float(brightness))
    return adjusted
