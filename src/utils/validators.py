"""Input validation utilities for image processing."""

from typing import Tuple, Union
import numpy as np


class ImageValidationError(ValueError):
    """Raised when an image fails validation checks."""
    pass


def validate_image_array(image: np.ndarray) -> None:
    """Validate that the input is a valid non-empty NumPy image array.
    
    Args:
        image: NumPy array representing an image.
        
    Raises:
        ImageValidationError: If the image is None, empty, or not a NumPy array.
    """
    if image is None:
        raise ImageValidationError("Image array cannot be None.")
    
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"Expected image as np.ndarray, got {type(image).__name__}.")
    
    if image.size == 0:
        raise ImageValidationError("Image array is empty (size 0).")
    
    if image.ndim not in (2, 3):
        raise ImageValidationError(f"Expected 2D (grayscale) or 3D (color) array, got {image.ndim}D.")
    
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ImageValidationError(f"Invalid image dimensions: height={h}, width={w}.")


def validate_kernel_size(kernel_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Validate that a kernel size is positive and odd.
    
    Args:
        kernel_size: Integer or (width, height) tuple.
        
    Returns:
        A tuple of two positive odd integers (kw, kh).
        
    Raises:
        ValueError: If kernel size is invalid, negative, or even.
    """
    if isinstance(kernel_size, int):
        kw, kh = kernel_size, kernel_size
    elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
        kw, kh = int(kernel_size[0]), int(kernel_size[1])
    else:
        raise ValueError("kernel_size must be an integer or a 2-tuple of integers.")
        
    if kw <= 0 or kh <= 0:
        raise ValueError(f"Kernel dimensions must be positive, got ({kw}, {kh}).")
        
    if kw % 2 == 0 or kh % 2 == 0:
        raise ValueError(f"Kernel dimensions must be odd integers, got ({kw}, {kh}).")
        
    return kw, kh
