"""Unit tests for Phase 4: Perspective Transformation."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.detection.document_detection import find_document_corners
from src.transformation.perspective import (
    compute_destination_dimensions,
    order_points,
    warp_perspective,
)


@pytest.fixture
def sample_corners():
    """Canonical rectangle corners: TL=(50, 50), TR=(250, 50), BR=(250, 350), BL=(50, 350)."""
    return np.array([
        [50, 50],
        [250, 50],
        [250, 350],
        [50, 350]
    ], dtype=np.float32)


@pytest.fixture
def sample_document_image():
    """Load sample document photograph."""
    path = Path("assets/sample_images/sample_document.png")
    if not path.exists():
        pytest.skip("Sample document image not generated.")
    return cv2.imread(str(path))


def test_order_points_shuffled(sample_corners):
    """Test order_points correctly canonicalizes any permutation of corners."""
    # Permute the 4 points
    shuffled_indices = [2, 0, 3, 1]
    shuffled = sample_corners[shuffled_indices]

    ordered = order_points(shuffled)

    # Check indices: TL=(50, 50), TR=(250, 50), BR=(250, 350), BL=(50, 350)
    assert np.allclose(ordered[0], [50, 50])    # Top-Left
    assert np.allclose(ordered[1], [250, 50])   # Top-Right
    assert np.allclose(ordered[2], [250, 350])  # Bottom-Right
    assert np.allclose(ordered[3], [50, 350])   # Bottom-Left


def test_order_points_invalid():
    """Test order_points rejects invalid inputs."""
    with pytest.raises(ValueError):
        order_points(None)
    with pytest.raises(ValueError):
        order_points(np.array([[10, 10], [20, 20]]))  # Only 2 points
    with pytest.raises(ValueError):
        order_points(np.zeros((5, 2)))  # 5 points


def test_compute_destination_dimensions(sample_corners):
    """Test calculation of target width and height on known rectangle."""
    # Width = 250 - 50 = 200; Height = 350 - 50 = 300
    width, height = compute_destination_dimensions(sample_corners)
    assert width == 200
    assert height == 300


def test_compute_destination_dimensions_trapezoid():
    """Test calculation takes the maximum opposite edge length for trapezoids."""
    trapezoid = np.array([
        [100, 50],   # TL
        [300, 50],   # TR: top width = 200
        [350, 350],  # BR: bottom width = 350 - 50 = 300
        [50, 350]    # BL
    ], dtype=np.float32)

    width, height = compute_destination_dimensions(trapezoid)
    assert width == 300   # max(200, 300)
    assert height == 304  # max(hypot(50, 300), hypot(50, 300)) = round(sqrt(92500)) = 304


def test_warp_perspective_basic():
    """Test perspective warping on a synthetic canvas."""
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    pts = np.array([[50, 50], [200, 40], [220, 250], [40, 260]], dtype=np.float32)

    warped = warp_perspective(canvas, pts)
    assert isinstance(warped, np.ndarray)
    assert warped.ndim == 3
    assert warped.shape[0] > 0
    assert warped.shape[1] > 0


def test_warp_perspective_with_explicit_size():
    """Test perspective warping with user-specified target dimensions."""
    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 200
    pts = np.array([[20, 20], [300, 20], [300, 400], [20, 400]], dtype=np.float32)

    warped = warp_perspective(canvas, pts, output_size=(600, 800))
    assert warped.shape == (800, 600, 3)  # height=800, width=600


def test_warp_perspective_on_sample_document(sample_document_image):
    """Test end-to-end detection and warping on the sample document photograph."""
    corners, success, _ = find_document_corners(sample_document_image)
    assert success is True

    warped = warp_perspective(sample_document_image, corners)

    # In our realistic invoice photograph, the warped output is a vertical/portrait document
    h, w = warped.shape[:2]
    aspect_ratio = w / float(h)
    expected_ratio = 942.0 / 1074.0  # ~0.877

    assert pytest.approx(aspect_ratio, abs=0.05) == expected_ratio
    assert h >= 1000
    assert w >= 900
