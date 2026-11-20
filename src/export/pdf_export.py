"""High-quality PDF document generation from processed scans."""

import io
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image
from reportlab.lib import pagesizes
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.utils.validators import validate_image_array

PAGE_SIZES = {
    "A4": pagesizes.A4,          # 595.27 x 841.89 points
    "LETTER": pagesizes.letter,  # 612.0 x 792.0 points
    "FIT": None,                 # Exact image dimensions
}


def export_image_to_pdf(
    image: np.ndarray,
    page_size: str = "A4",
    margin_pt: float = 18.0
) -> bytes:
    """Generate a high-quality PDF document preserving aspect ratio.
    
    Fits the scanned document centered on a standard page (e.g. A4, Letter) with
    consistent margins, or creates a custom page size matching the image aspect ratio.
    
    Args:
        image: Scanned document array (Grayscale, BGR, or binary uint8).
        page_size: Target page format ('A4', 'LETTER', or 'FIT').
        margin_pt: Margin size in typographic points (72 pt = 1 inch, default 18 pt = 0.25 in).
        
    Returns:
        Bytes sequence containing the complete PDF file.
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If page_size is invalid.
    """
    validate_image_array(image)
    size_upper = page_size.upper()
    if size_upper not in PAGE_SIZES:
        raise ValueError(
            f"Unsupported page_size '{page_size}'. Supported: {', '.join(sorted(PAGE_SIZES.keys()))}"
        )

    # Standardize to RGB or Grayscale PIL Image
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    else:
        gray = image if image.ndim == 2 else image[:, :, 0]
        pil_img = Image.fromarray(gray)

    img_w, img_h = pil_img.size
    img_aspect = img_w / float(img_h)

    # Convert PIL Image to buffer for ReportLab ImageReader
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    reader = ImageReader(img_buffer)

    pdf_buffer = io.BytesIO()

    if size_upper == "FIT":
        # Custom page size exactly matching image dimensions at 72 DPI
        page_w = img_w
        page_h = img_h
        c = canvas.Canvas(pdf_buffer, pagesize=(page_w, page_h))
        c.drawImage(reader, 0, 0, width=page_w, height=page_h)
        c.showPage()
        c.save()
    else:
        # Standard page (A4 or Letter) with margin
        std_page = PAGE_SIZES[size_upper]
        page_w, page_h = std_page

        usable_w = page_w - (2 * margin_pt)
        usable_h = page_h - (2 * margin_pt)

        # Compute scaling to fit without distorting aspect ratio
        scale_w = usable_w / float(img_w)
        scale_h = usable_h / float(img_h)
        scale = min(scale_w, scale_h)

        draw_w = img_w * scale
        draw_h = img_h * scale

        # Center on the page
        x_offset = margin_pt + (usable_w - draw_w) / 2.0
        y_offset = margin_pt + (usable_h - draw_h) / 2.0

        c = canvas.Canvas(pdf_buffer, pagesize=(page_w, page_h))
        c.drawImage(reader, x_offset, y_offset, width=draw_w, height=draw_h)
        c.showPage()
        c.save()

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
