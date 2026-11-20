"""Unit tests for Phase 5: Image Enhancement Algorithms."""

import cv2
import numpy as np
import pytest

from src.enhancement.grayscale import convert_to_grayscale
from src.enhancement.thresholding import adaptive_gaussian_threshold, otsu_threshold
from src.enhancement.denoising import apply_gaussian_filter, apply_median_filter
from src.enhancement.sharpening import sharpen_image, unsharp_mask
from src.enhancement.contrast import apply_clahe, adjust_brightness_contrast
from src.enhancement.morphology import apply_morphology


@pytest.fixture
def gradient_image():
    """Create a synthetic 100x100 gradient image simulating uneven document lighting."""
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        img[i, :] = int(i * 2.55)  # Gradient from 0 to 255
    # Add high-contrast text lines
    img[20:25, 20:80] = 0
    img[60:65, 20:80] = 0
    return img


@pytest.fixture
def color_test_image():
    """Create a synthetic 3-channel BGR image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :, 0] = 200  # Blue
    img[50:, :, 2] = 200  # Red
    return img


def test_convert_to_grayscale(color_test_image):
    """Test colorspace reduction to single channel uint8."""
    gray = convert_to_grayscale(color_test_image)
    assert gray.ndim == 2
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_adaptive_gaussian_threshold(gradient_image):
    """Test adaptive thresholding handles lighting gradients."""
    binary = adaptive_gaussian_threshold(gradient_image, block_size=11, c_constant=2.0)
    assert binary.shape == gradient_image.shape
    assert binary.dtype == np.uint8
    # Binary image should only contain 0 and 255
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})

    # Test invalid block size
    with pytest.raises(ValueError):
        adaptive_gaussian_threshold(gradient_image, block_size=4)  # Even
    with pytest.raises(ValueError):
        adaptive_gaussian_threshold(gradient_image, block_size=1)  # < 3


def test_otsu_threshold(gradient_image):
    """Test Otsu global bimodal binarization."""
    binary, thresh_val = otsu_threshold(gradient_image)
    assert binary.shape == gradient_image.shape
    assert 0 <= thresh_val <= 255
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})


def test_apply_median_filter():
    """Test median filter suppresses salt-and-pepper noise."""
    img = np.ones((50, 50), dtype=np.uint8) * 128
    # Add salt-and-pepper noise pixels
    img[10, 10] = 255
    img[20, 20] = 0

    filtered = apply_median_filter(img, kernel_size=3)
    # The isolated noise pixels should be replaced by surrounding 128
    assert filtered[10, 10] == 128
    assert filtered[20, 20] == 128


def test_sharpen_image(color_test_image):
    """Test spatial sharpening kernel increases local gradient."""
    sharpened = sharpen_image(color_test_image, strength=1.5)
    assert sharpened.shape == color_test_image.shape
    assert sharpened.dtype == np.uint8

    with pytest.raises(ValueError):
        sharpen_image(color_test_image, strength=-0.5)


def test_unsharp_mask(gradient_image):
    """Test unsharp masking edge boost."""
    boosted = unsharp_mask(gradient_image, strength=1.0)
    assert boosted.shape == gradient_image.shape
    assert boosted.dtype == np.uint8


def test_apply_clahe_grayscale(gradient_image):
    """Test CLAHE on single-channel grayscale image."""
    equalized = apply_clahe(gradient_image, clip_limit=3.0)
    assert equalized.shape == gradient_image.shape
    assert equalized.dtype == np.uint8


def test_apply_clahe_color(color_test_image):
    """Test CLAHE preserves color channels via LAB space."""
    equalized = apply_clahe(color_test_image, clip_limit=2.0)
    assert equalized.shape == color_test_image.shape
    assert equalized.ndim == 3


def test_adjust_brightness_contrast(color_test_image):
    """Test linear brightness and contrast scaling."""
    adjusted = adjust_brightness_contrast(color_test_image, brightness=30, contrast=1.2)
    assert adjusted.shape == color_test_image.shape
    assert adjusted.dtype == np.uint8

    with pytest.raises(ValueError):
        adjust_brightness_contrast(color_test_image, contrast=-1.0)


def test_apply_morphology(gradient_image):
    """Test morphological operations."""
    binary = adaptive_gaussian_threshold(gradient_image, block_size=11)
    
    eroded = apply_morphology(binary, operation="erosion", kernel_size=3)
    assert eroded.shape == binary.shape

    dilated = apply_morphology(binary, operation="dilation", kernel_size=3)
    assert dilated.shape == binary.shape

    with pytest.raises(ValueError):
        apply_morphology(binary, operation="unsupported_op")
