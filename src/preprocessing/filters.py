"""Basic image filtering and colorspace conversion functions."""

from typing import Tuple, Union
import cv2
import numpy as np

from src.utils.validators import validate_image_array, validate_kernel_size


def to_grayscale(image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """Convert an image to single-channel 8-bit grayscale.
    
    If the image is already single-channel, a copy is returned.
    
    Args:
        image: Source image as NumPy ndarray.
        is_bgr: True if 3-channel input is in BGR format (OpenCV standard),
               False if RGB.
               
    Returns:
        Single-channel grayscale image array (uint8).
        
    Raises:
        ImageValidationError: If the input array is invalid.
    """
    validate_image_array(image)

    if image.ndim == 2:
        return image.copy()

    if image.shape[2] == 1:
        return image[:, :, 0].copy()

    code = cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
    return cv2.cvtColor(image, code)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = (5, 5),
    sigma_x: float = 0.0,
    sigma_y: float = 0.0
) -> np.ndarray:
    """Apply Gaussian blur to smooth the image and suppress high-frequency noise.
    
    The Gaussian filter convolves the image with a 2D Gaussian kernel:
    G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    
    Args:
        image: Input image (grayscale or color).
        kernel_size: Kernel dimensions (kw, kh) or single int (k, k). Must be positive and odd.
        sigma_x: Gaussian kernel standard deviation in X direction (0 = computed from kernel size).
        sigma_y: Gaussian kernel standard deviation in Y direction (0 = defaults to sigma_x).
        
    Returns:
        Blurred image with identical dimensions and type.
        
    Raises:
        ImageValidationError: If the input array is invalid.
        ValueError: If kernel dimensions are negative, zero, or even numbers.
    """
    validate_image_array(image)
    kw, kh = validate_kernel_size(kernel_size)
    return cv2.GaussianBlur(image, (kw, kh), sigmaX=sigma_x, sigmaY=sigma_y)
