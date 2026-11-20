"""Morphological filtering operations for binary and grayscale documents."""

from typing import Tuple, Union
import cv2
import numpy as np

from src.utils.validators import validate_image_array, validate_kernel_size

MORPH_OPERATIONS = {
    "erosion": cv2.MORPH_ERODE,
    "dilation": cv2.MORPH_DILATE,
    "opening": cv2.MORPH_OPEN,
    "closing": cv2.MORPH_CLOSE,
}


def apply_morphology(
    image: np.ndarray,
    operation: str = "closing",
    kernel_size: Union[int, Tuple[int, int]] = (3, 3),
    iterations: int = 1
) -> np.ndarray:
    """Apply morphological transforms to clean up binary scans or text masks.
    
    Operations:
    - erosion: Thins bright foreground strokes; removes tiny white pixel noise.
    - dilation: Thickens bright strokes; bridges small breaks in character lines.
    - opening (erosion then dilation): Eliminates small background noise specs.
    - closing (dilation then erosion): Fills small holes and cracks in dark letters.
    
    Args:
        image: Source image array (Grayscale or binary uint8).
        operation: One of {'erosion', 'dilation', 'opening', 'closing'}.
        kernel_size: Dimensions of the structuring element (must be odd positive integers).
        iterations: Number of application passes (default 1).
        
    Returns:
        Morphologically filtered image array.
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If operation name or parameters are invalid.
    """
    validate_image_array(image)
    op_lower = operation.lower()
    if op_lower not in MORPH_OPERATIONS:
        raise ValueError(
            f"Unsupported morphology operation '{operation}'. Supported: {', '.join(sorted(MORPH_OPERATIONS.keys()))}"
        )
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}.")

    kw, kh = validate_kernel_size(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    op_code = MORPH_OPERATIONS[op_lower]

    return cv2.morphologyEx(image, op_code, kernel, iterations=iterations)
