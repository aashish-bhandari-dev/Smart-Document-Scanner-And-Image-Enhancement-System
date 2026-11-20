"""Binarization algorithms: Adaptive Gaussian and Otsu thresholding."""

from typing import Tuple, Union
import cv2
import numpy as np

from src.enhancement.grayscale import convert_to_grayscale
from src.utils.validators import validate_image_array


def adaptive_gaussian_threshold(
    image: np.ndarray,
    block_size: int = 15,
    c_constant: float = 8.0,
    invert: bool = False
) -> np.ndarray:
    """Apply Adaptive Gaussian Thresholding to produce a clean black-and-white document.
    
    Rather than using a single global threshold value across the entire image, adaptive
    thresholding calculates a dynamic threshold T(x, y) for each individual pixel based
    on a Gaussian-weighted neighborhood of size (block_size x block_size) minus constant C:
    
    T(x, y) = GaussianWeightedMean(neighborhood) - C
    
    This is ideal for photographed documents with shadows or lighting gradients across
    the paper.
    
    Args:
        image: Input image (grayscale or color BGR).
        block_size: Size of pixel neighborhood. Must be an odd integer >= 3.
        c_constant: Constant subtracted from the calculated mean. Fine-tunes sensitivity.
        invert: If True, produces white text on black background.
        
    Returns:
        Binary image array (uint8) with values 0 (black) and 255 (white).
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If block_size is even or < 3.
    """
    validate_image_array(image)
    if block_size < 3:
        raise ValueError(f"block_size must be >= 3, got {block_size}.")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be an odd integer, got {block_size}.")

    gray = convert_to_grayscale(image)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=thresh_type,
        blockSize=block_size,
        C=float(c_constant)
    )

    return binary


def otsu_threshold(
    image: np.ndarray,
    pre_blur_ksize: int = 0,
    invert: bool = False
) -> Tuple[np.ndarray, float]:
    """Binarize an image using Otsu's optimal global clustering threshold.
    
    Otsu's method treats the grayscale intensity histogram as a bimodal distribution
    (foreground text and background paper) and computes the threshold value that maximizes
    the between-class variance:
    
    sigma_B^2 = omega_0 * omega_1 * (mu_0 - mu_1)^2
    
    Args:
        image: Input image (grayscale or color BGR).
        pre_blur_ksize: Optional Gaussian blur kernel size (e.g. 3 or 5) prior to thresholding.
        invert: If True, produces white text on black background.
        
    Returns:
        Tuple of (binary_image: np.ndarray, optimal_threshold_value: float).
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If pre_blur_ksize is invalid.
    """
    validate_image_array(image)
    gray = convert_to_grayscale(image)

    if pre_blur_ksize > 0:
        if pre_blur_ksize % 2 == 0:
            raise ValueError(f"pre_blur_ksize must be odd, got {pre_blur_ksize}.")
        gray = cv2.GaussianBlur(gray, (pre_blur_ksize, pre_blur_ksize), 0)

    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    if invert:
        flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    opt_thresh, binary = cv2.threshold(gray, 0, 255, flags)
    return binary, float(opt_thresh)
