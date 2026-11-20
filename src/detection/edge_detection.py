"""Edge detection algorithms and morphological edge closing."""

from typing import Optional, Tuple, Union
import cv2
import numpy as np

from src.utils.validators import validate_image_array, validate_kernel_size


def detect_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = True
) -> np.ndarray:
    """Detect edges using the Canny algorithm.
    
    The Canny detector runs in four stages:
    1. Smooth the image with a Gaussian filter.
    2. Compute gradient magnitude and direction using Sobel operators.
    3. Apply Non-Maximum Suppression (NMS) to thin edge ridges.
    4. Apply Hysteresis thresholding with low and high thresholds.
    
    Args:
        image: Single-channel grayscale image (uint8).
        low_threshold: Lower bound for hysteresis (weak edges).
        high_threshold: Upper bound for hysteresis (strong edges).
        aperture_size: Sobel kernel aperture size (3, 5, or 7).
        l2_gradient: Whether to use L2 norm (sqrt(Gx^2 + Gy^2)) or L1 norm (|Gx| + |Gy|).
        
    Returns:
        Binary edge map (2D uint8 ndarray) where 255 denotes edge pixels.
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If thresholds or aperture size are invalid.
    """
    validate_image_array(image)
    if image.ndim != 2:
        raise ValueError(f"detect_edges expects a 2D grayscale image, got shape {image.shape}.")
    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Canny thresholds must be non-negative.")
    if low_threshold > high_threshold:
        raise ValueError(f"low_threshold ({low_threshold}) cannot exceed high_threshold ({high_threshold}).")
    if aperture_size not in (3, 5, 7):
        raise ValueError(f"aperture_size must be 3, 5, or 7, got {aperture_size}.")

    return cv2.Canny(
        image,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient
    )


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Compute optimal Canny edge thresholds automatically based on image median intensity.
    
    T_lower = max(0, (1.0 - sigma) * median)
    T_upper = min(255, (1.0 + sigma) * median)
    
    Args:
        image: Single-channel grayscale image.
        sigma: Spread factor around the median intensity (default 0.33).
        
    Returns:
        Binary edge map.
    """
    validate_image_array(image)
    if image.ndim != 2:
        raise ValueError(f"auto_canny expects a 2D grayscale image, got shape {image.shape}.")

    median_val = np.median(image)
    low_threshold = int(max(0, (1.0 - sigma) * median_val))
    high_threshold = int(min(255, (1.0 + sigma) * median_val))

    # Guard against identical or collapsed thresholds
    if low_threshold == high_threshold:
        low_threshold = max(0, low_threshold - 20)
        high_threshold = min(255, high_threshold + 20)

    return detect_edges(image, low_threshold=low_threshold, high_threshold=high_threshold)


def close_edges(
    edge_map: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = (5, 5),
    iterations: int = 1
) -> np.ndarray:
    """Apply morphological closing (dilation followed by erosion) to bridge broken edge lines.
    
    In physical document photographs, paper creases, light reflections, or low-contrast
    desk boundaries can fracture the document outline. Morphological closing fills
    small gaps and connects perimeter segments.
    
    Args:
        edge_map: Binary edge image (uint8).
        kernel_size: Structuring element size.
        iterations: Number of times to apply closing.
        
    Returns:
        Closed binary edge map.
    """
    validate_image_array(edge_map)
    kw, kh = validate_kernel_size(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    return cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=iterations)
