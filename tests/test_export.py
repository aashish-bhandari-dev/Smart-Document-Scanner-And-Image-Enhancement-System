"""Unit tests for Phase 7: Document Export Subsystem."""

import numpy as np
import pytest

from src.export.image_export import export_image_as_bytes
from src.export.pdf_export import export_image_to_pdf


@pytest.fixture
def sample_scanned_image():
    """Create a synthetic 600x400 document image."""
    img = np.ones((600, 400, 3), dtype=np.uint8) * 240
    # Add a border
    img[10:590, 10:390] = 255
    return img


def test_export_png(sample_scanned_image):
    """Test exporting image as PNG bytes."""
    data, mime = export_image_as_bytes(sample_scanned_image, format="PNG")
    assert isinstance(data, bytes)
    assert len(data) > 0
    assert mime == "image/png"
    # PNG magic number: \x89PNG\r\n\x1a\n
    assert data.startswith(b"\x89PNG")


def test_export_jpeg(sample_scanned_image):
    """Test exporting image as JPEG bytes."""
    data, mime = export_image_as_bytes(sample_scanned_image, format="JPEG", quality=90)
    assert isinstance(data, bytes)
    assert len(data) > 0
    assert mime == "image/jpeg"
    # JPEG magic number: \xff\xd8
    assert data.startswith(b"\xff\xd8")


def test_export_invalid_format(sample_scanned_image):
    """Test unsupported format raises ValueError."""
    with pytest.raises(ValueError):
        export_image_as_bytes(sample_scanned_image, format="GIF")


def test_export_invalid_quality(sample_scanned_image):
    """Test out-of-range quality raises ValueError."""
    with pytest.raises(ValueError):
        export_image_as_bytes(sample_scanned_image, format="JPEG", quality=150)


def test_export_pdf_a4(sample_scanned_image):
    """Test generating A4 PDF document."""
    pdf_bytes = export_image_to_pdf(sample_scanned_image, page_size="A4")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    # PDF files start with %PDF-
    assert pdf_bytes.startswith(b"%PDF-")


def test_export_pdf_letter(sample_scanned_image):
    """Test generating Letter PDF document."""
    pdf_bytes = export_image_to_pdf(sample_scanned_image, page_size="LETTER")
    assert pdf_bytes.startswith(b"%PDF-")


def test_export_pdf_fit(sample_scanned_image):
    """Test generating exact-fit PDF document."""
    pdf_bytes = export_image_to_pdf(sample_scanned_image, page_size="FIT")
    assert pdf_bytes.startswith(b"%PDF-")


def test_export_pdf_invalid_page_size(sample_scanned_image):
    """Test invalid page size raises ValueError."""
    with pytest.raises(ValueError):
        export_image_to_pdf(sample_scanned_image, page_size="LEGAL_UNSUPPORTED")
