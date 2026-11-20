"""Noise reduction and smoothing filters."""

from typing import Tuple, Union
import cv2
import numpy as np

from src.utils.validators import validate_image_array, validate_kernel_size


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = (5, 5),
    sigma_x: float = 0.0
) -> np.ndarray:
    """Reduce high-frequency sensor noise using 2D Gaussian filtering.
    
    Args:
        image: Source image array (Grayscale or Color).
        kernel_size: Kernel dimensions (kw, kh) or odd integer k.
        sigma_x: Standard deviation along X axis (0 computes from kernel size).
        
    Returns:
        Filtered smoothed image.
    """
    validate_image_array(image)
    kw, kh = validate_kernel_size(kernel_size)
    return cv2.GaussianBlur(image, (kw, kh), sigmaX=sigma_x)


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """Remove impulse (salt-and-pepper) noise using non-linear median filtering.
    
    Unlike linear smoothing filters that blur sharp text boundaries, the median filter
    replaces each pixel with the median value of its neighborhood. This completely
    eradicates isolated speckles and ink spatters while maintaining crisp character strokes.
    
    Args:
        image: Source image array (Grayscale or Color).
        kernel_size: Neighborhood diameter. Must be an odd positive integer >= 1.
        
    Returns:
        Median-filtered image.
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If kernel_size is even or < 1.
    """
    validate_image_array(image)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"Median filter kernel_size must be an odd positive integer, got {kernel_size}.")

    if kernel_size == 1:
        return image.copy()

    return cv2.medianBlur(image, kernel_size)
