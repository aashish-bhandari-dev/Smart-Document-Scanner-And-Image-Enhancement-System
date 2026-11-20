"""Grayscale conversion algorithms and color channel utilities."""

import cv2
import numpy as np

from src.utils.validators import validate_image_array


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image array (BGR or RGB) to single-channel 8-bit grayscale.
    
    Applies standard ITU-R BT.601 luminance weights:
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    
    If the image is already single-channel, returns a defensive copy.
    
    Args:
        image: Source image array as NumPy ndarray.
        
    Returns:
        Single-channel grayscale image (uint8).
        
    Raises:
        ImageValidationError: If image is invalid.
    """
    validate_image_array(image)

    if image.ndim == 2:
        return image.copy()

    if image.shape[2] == 1:
        return image[:, :, 0].copy()

    if image.shape[2] == 4:
        # RGBA / BGRA -> Gray
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
