"""Text and edge sharpening algorithms."""

import cv2
import numpy as np

from src.utils.validators import validate_image_array


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Enhance text clarity and character outlines using a spatial sharpening kernel.
    
    Constructs an adjustable high-pass boost kernel:
    K = [[ 0,         -strength/4,   0        ],
         [-strength/4, 1 + strength, -strength/4],
         [ 0,         -strength/4,   0        ]]
    
    Args:
        image: Source image array (Grayscale or Color).
        strength: Sharpening intensity factor >= 0.0 (typically 0.5 - 2.0).
        
    Returns:
        Sharpened image with uint8 clipping.
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If strength is negative.
    """
    validate_image_array(image)
    if strength < 0.0:
        raise ValueError(f"Sharpening strength must be non-negative, got {strength}.")

    if strength == 0.0:
        return image.copy()

    s = float(strength)
    kernel = np.array([
        [0.0, -s / 4.0, 0.0],
        [-s / 4.0, 1.0 + s, -s / 4.0],
        [0.0, -s / 4.0, 0.0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.5,
    threshold: int = 0
) -> np.ndarray:
    """Sharpen text using unsharp masking.
    
    Computes a high-frequency difference mask by subtracting a Gaussian-smoothed
    version from the original, and adds back the boosted mask:
    
    detail_mask = image - GaussianBlur(image)
    output = image + strength * detail_mask
    
    Args:
        image: Input image array (Grayscale or Color).
        sigma: Gaussian blur standard deviation.
        strength: Edge amplification weight >= 0.
        threshold: Minimum pixel difference required to apply boost (filters flat noise).
        
    Returns:
        Sharpened image array.
    """
    validate_image_array(image)
    if strength <= 0.0:
        return image.copy()

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    mask = cv2.subtract(image, blurred)

    if threshold > 0:
        # Suppress noise by ignoring differences below threshold
        low_contrast_mask = np.abs(mask) < threshold
        mask[low_contrast_mask] = 0

    boosted = cv2.addWeighted(image, 1.0, mask, float(strength), 0)
    return np.clip(boosted, 0, 255).astype(np.uint8)
