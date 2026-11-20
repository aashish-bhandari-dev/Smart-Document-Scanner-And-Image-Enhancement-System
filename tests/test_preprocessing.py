"""Unit tests for Phase 2: Preprocessing and Image Loading."""

import io
import cv2
import numpy as np
import pytest
from PIL import Image

from src.preprocessing.image_loader import load_image
from src.preprocessing.resize import resize_image
from src.preprocessing.filters import to_grayscale, apply_gaussian_blur
from src.utils.validators import ImageValidationError


@pytest.fixture
def sample_rgb_image():
    """Create a synthetic 100x200 BGR test image."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:50, :, 0] = 255  # Blue upper half
    img[50:, :, 2] = 255  # Red lower half
    return img


@pytest.fixture
def sample_png_bytes(sample_rgb_image):
    """Encode synthetic image as PNG bytes."""
    success, buffer = cv2.imencode(".png", sample_rgb_image)
    assert success
    return buffer.tobytes()


def test_load_image_from_bytes(sample_png_bytes):
    """Test loading valid image from raw bytes."""
    img, meta = load_image(sample_png_bytes)
    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 200, 3)
    assert meta["width"] == 200
    assert meta["height"] == 100
    assert meta["channels"] == 3
    assert meta["format"] == "PNG"
    assert meta["aspect_ratio"] == 2.0


def test_load_image_empty_bytes():
    """Test loading empty bytes raises ImageValidationError."""
    with pytest.raises(ImageValidationError):
        load_image(b"")


def test_load_image_corrupt_data():
    """Test loading corrupt byte data raises ImageValidationError."""
    with pytest.raises(ImageValidationError):
        load_image(b"not_an_image_file_content")


def test_load_image_file_like_object(sample_png_bytes):
    """Test loading from a file-like buffer (e.g., Streamlit UploadedFile)."""
    buf = io.BytesIO(sample_png_bytes)
    setattr(buf, "name", "document.png")
    img, meta = load_image(buf)
    assert img.shape == (100, 200, 3)
    assert meta["filename"] == "document.png"


def test_resize_image_by_height(sample_rgb_image):
    """Test resizing to a specific height while maintaining aspect ratio."""
    resized, ratio = resize_image(sample_rgb_image, height=50)
    assert resized.shape[0] == 50
    assert resized.shape[1] == 100
    assert pytest.approx(ratio, 0.01) == 2.0


def test_resize_image_by_max_dimension(sample_rgb_image):
    """Test resizing by max_dimension constraint."""
    resized, ratio = resize_image(sample_rgb_image, max_dimension=100)
    assert max(resized.shape[:2]) == 100
    assert resized.shape == (50, 100, 3)
    assert pytest.approx(ratio, 0.01) == 2.0


def test_resize_image_invalid_dimensions(sample_rgb_image):
    """Test invalid resize parameters."""
    with pytest.raises(ValueError):
        resize_image(sample_rgb_image, height=-10)
    with pytest.raises(ValueError):
        resize_image(sample_rgb_image, max_dimension=0)


def test_to_grayscale(sample_rgb_image):
    """Test converting BGR image to single-channel grayscale."""
    gray = to_grayscale(sample_rgb_image, is_bgr=True)
    assert gray.ndim == 2
    assert gray.shape == (100, 200)
    assert gray.dtype == np.uint8

    # Passing already grayscale image should return identical copy
    gray_again = to_grayscale(gray)
    assert gray_again.shape == gray.shape
    assert np.array_equal(gray, gray_again)


def test_apply_gaussian_blur(sample_rgb_image):
    """Test Gaussian blur filtering."""
    blurred = apply_gaussian_blur(sample_rgb_image, kernel_size=(5, 5))
    assert blurred.shape == sample_rgb_image.shape
    assert blurred.dtype == sample_rgb_image.dtype

    # Test single integer kernel size
    blurred_int = apply_gaussian_blur(sample_rgb_image, kernel_size=3)
    assert blurred_int.shape == sample_rgb_image.shape

    # Even kernel sizes must be rejected
    with pytest.raises(ValueError):
        apply_gaussian_blur(sample_rgb_image, kernel_size=(4, 4))
    with pytest.raises(ValueError):
        apply_gaussian_blur(sample_rgb_image, kernel_size=-3)
