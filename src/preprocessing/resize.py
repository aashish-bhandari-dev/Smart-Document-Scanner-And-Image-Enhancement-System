"""Image resizing with aspect ratio preservation."""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.validators import validate_image_array


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_dimension: Optional[int] = None,
    interpolation: int = cv2.INTER_AREA
) -> Tuple[np.ndarray, float]:
    """Resize an image while preserving aspect ratio.
    
    Computes and returns the downscaled/upscaled image along with the scaling ratio
    (original_dimension / resized_dimension). This ratio allows scaling coordinates
    detected on the small image back to the original full-resolution image.
    
    Args:
        image: Source image as NumPy ndarray (BGR or Grayscale).
        width: Target width in pixels (optional).
        height: Target height in pixels (optional).
        max_dimension: Target maximum dimension for either width or height (optional).
        interpolation: OpenCV interpolation method (cv2.INTER_AREA is best for shrinking).
        
    Returns:
        Tuple of (resized_image, scale_ratio).
        scale_ratio = original_height / resized_height (or 1.0 if not resized).
        
    Raises:
        ImageValidationError: If input image is invalid.
        ValueError: If dimension parameters are negative or zero.
    """
    validate_image_array(image)
    orig_h, orig_w = image.shape[:2]

    if max_dimension is not None:
        if max_dimension <= 0:
            raise ValueError(f"max_dimension must be positive, got {max_dimension}.")
        if max(orig_h, orig_w) > max_dimension:
            if orig_h >= orig_w:
                height = max_dimension
                width = None
            else:
                width = max_dimension
                height = None
        else:
            # Image is already within max_dimension
            return image.copy(), 1.0

    if width is None and height is None:
        return image.copy(), 1.0

    if width is not None and width <= 0:
        raise ValueError(f"Target width must be positive, got {width}.")
    if height is not None and height <= 0:
        raise ValueError(f"Target height must be positive, got {height}.")

    if width is None and height is not None:
        ratio = orig_h / float(height)
        new_w = int(round(orig_w / ratio))
        new_h = height
    elif height is None and width is not None:
        ratio = orig_w / float(width)
        new_h = int(round(orig_h / ratio))
        new_w = width
    else:
        # Both width and height specified explicitly
        assert width is not None and height is not None
        new_w, new_h = width, height
        ratio = orig_h / float(height)

    # Ensure minimum 1 pixel
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Choose optimal interpolation: INTER_AREA for downsampling, INTER_CUBIC/LINEAR for upsampling
    if new_w < orig_w or new_h < orig_h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    return resized, ratio
