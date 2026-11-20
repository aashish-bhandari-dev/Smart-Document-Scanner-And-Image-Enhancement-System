"""Image loading, format validation, and metadata extraction."""

import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.utils.validators import ImageValidationError, validate_image_array

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}


def load_image(
    source: Union[str, Path, bytes, BinaryIO],
    max_dimension: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load and validate an image from a path, raw bytes, or file-like object.
    
    Corrects EXIF orientation, validates formats, and returns a BGR image array
    along with comprehensive image metadata.
    
    Args:
        source: File path (str/Path), byte sequence, or binary file-like object.
        max_dimension: Optional maximum dimension (width or height) to clamp to.
        
    Returns:
        Tuple of (bgr_image_array, metadata_dictionary).
        
    Raises:
        ImageValidationError: If the image cannot be read, is corrupted, or has an unsupported format.
        FileNotFoundError: If a provided string path does not exist.
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix.lower() not in SUPPORTED_FORMATS:
                raise ImageValidationError(
                    f"Unsupported format '{path.suffix}'. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
                )
            pil_image = Image.open(path)
            file_size_bytes = path.stat().st_size
            filename = path.name
        elif isinstance(source, bytes):
            if len(source) == 0:
                raise ImageValidationError("Image byte stream is empty.")
            pil_image = Image.open(io.BytesIO(source))
            file_size_bytes = len(source)
            filename = "memory_buffer"
        else:
            # File-like object (e.g., Streamlit's UploadedFile or BytesIO)
            source.seek(0)
            raw_bytes = source.read()
            if len(raw_bytes) == 0:
                raise ImageValidationError("Image buffer is empty.")
            pil_image = Image.open(io.BytesIO(raw_bytes))
            file_size_bytes = len(raw_bytes)
            filename = getattr(source, "name", "uploaded_file")
            # Check extension if filename has one
            suffix = Path(filename).suffix.lower()
            if suffix and suffix not in SUPPORTED_FORMATS:
                raise ImageValidationError(
                    f"Unsupported format '{suffix}'. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
                )

        # Verify integrity
        pil_image.verify()
        
        # Re-open after verify() because verify() clears image state in Pillow
        if isinstance(source, (str, Path)):
            pil_image = Image.open(source)
        elif isinstance(source, bytes):
            pil_image = Image.open(io.BytesIO(source))
        else:
            source.seek(0)
            pil_image = Image.open(io.BytesIO(source.read()))

        # Record format before transformations
        orig_format = pil_image.format or "UNKNOWN"

        # Correct camera rotation using EXIF orientation tag
        pil_image = ImageOps.exif_transpose(pil_image)

        # Convert to RGB (standardizing RGBA / Grayscale / Palette to 3 channels)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert PIL to NumPy BGR (OpenCV format)
        rgb_array = np.array(pil_image)
        bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        validate_image_array(bgr_image)
        
        height, width = bgr_image.shape[:2]
        metadata = {
            "filename": filename,
            "format": orig_format,
            "width": width,
            "height": height,
            "channels": 3,
            "file_size_bytes": file_size_bytes,
            "aspect_ratio": round(width / float(height), 3),
        }

        return bgr_image, metadata

    except (ImageValidationError, FileNotFoundError):
        raise
    except Exception as exc:
        raise ImageValidationError(f"Failed to decode image: {str(exc)}") from exc
