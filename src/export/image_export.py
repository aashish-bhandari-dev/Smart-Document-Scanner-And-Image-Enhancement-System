"""Image serialization and bytes export for JPEG and PNG formats."""

import io
from typing import Tuple
import cv2
import numpy as np
from PIL import Image

from src.utils.validators import validate_image_array

MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
}


def export_image_as_bytes(
    image: np.ndarray,
    format: str = "PNG",
    quality: int = 95
) -> Tuple[bytes, str]:
    """Encode an image array into downloadable bytes.
    
    Args:
        image: Image array (Grayscale, BGR, or binary uint8).
        format: Target format ('PNG' or 'JPEG').
        quality: JPEG compression quality (1 to 100). Ignored for PNG.
        
    Returns:
        Tuple of (image_bytes, mime_type_string).
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If format is unsupported or quality is out of range.
    """
    validate_image_array(image)
    fmt_upper = format.upper()
    if fmt_upper not in MIME_TYPES:
        raise ValueError(f"Unsupported export format '{format}'. Supported: PNG, JPEG.")

    if not (1 <= quality <= 100):
        raise ValueError(f"quality must be between 1 and 100, got {quality}.")

    # Convert BGR to RGB for Pillow if 3 channels
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    else:
        # Grayscale / Binary
        gray = image if image.ndim == 2 else image[:, :, 0]
        pil_img = Image.fromarray(gray)

    buffer = io.BytesIO()
    if fmt_upper in ("JPEG", "JPG"):
        # JPEG requires RGB mode (cannot save RGBA)
        if pil_img.mode != "RGB" and pil_img.mode != "L":
            pil_img = pil_img.convert("RGB")
        pil_img.save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        pil_img.save(buffer, format="PNG", optimize=True)

    buffer.seek(0)
    data = buffer.getvalue()
    mime = MIME_TYPES[fmt_upper]

    return data, mime
