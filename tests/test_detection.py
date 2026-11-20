"""Unit tests for Phase 3 and Phase 8: Document Boundary Detection and Edge Cases."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.detection.edge_detection import auto_canny, close_edges, detect_edges
from src.detection.contour_detection import approximate_polygon, find_and_sort_contours
from src.detection.document_detection import (
    compute_quadrilateral_angles,
    draw_document_boundary,
    find_document_corners,
    get_default_fallback_corners,
    is_valid_document_quadrilateral,
)


@pytest.fixture
def synthetic_rect_image():
    """Create a 400x500 image with a bright contrasting rectangle inside."""
    img = np.zeros((400, 500), dtype=np.uint8)
    # Draw a 200x300 white rectangle in the center
    cv2.rectangle(img, (100, 80), (400, 320), 255, -1)
    return img


@pytest.fixture
def sample_document_path():
    """Path to the generated sample document image."""
    return Path("assets/sample_images/sample_document.png")


def test_detect_edges_valid(synthetic_rect_image):
    """Test Canny edge detection identifies boundary pixels."""
    edges = detect_edges(synthetic_rect_image, low_threshold=50, high_threshold=150)
    assert edges.shape == synthetic_rect_image.shape
    assert edges.dtype == np.uint8
    assert np.count_nonzero(edges) > 0


def test_detect_edges_invalid_params(synthetic_rect_image):
    """Test validation of Canny thresholds."""
    with pytest.raises(ValueError):
        detect_edges(synthetic_rect_image, low_threshold=200, high_threshold=100)
    with pytest.raises(ValueError):
        detect_edges(synthetic_rect_image, low_threshold=-10, high_threshold=100)


def test_auto_canny(synthetic_rect_image):
    """Test auto_canny calculates dynamic thresholds without error."""
    edges = auto_canny(synthetic_rect_image)
    assert edges.shape == synthetic_rect_image.shape
    assert np.count_nonzero(edges) > 0


def test_close_edges():
    """Test morphological closing bridges broken line gaps."""
    edge_map = np.zeros((50, 50), dtype=np.uint8)
    edge_map[25, 10:20] = 255
    edge_map[25, 23:35] = 255  # Gap between 20 and 23

    closed = close_edges(edge_map, kernel_size=(5, 5))
    assert closed[25, 21] == 255


def test_find_and_sort_contours(synthetic_rect_image):
    """Test contour extraction and area sorting."""
    edges = detect_edges(synthetic_rect_image)
    contours = find_and_sort_contours(edges)
    assert len(contours) > 0
    areas = [cv2.contourArea(c) for c in contours]
    assert areas == sorted(areas, reverse=True)


def test_approximate_polygon_rectangle(synthetic_rect_image):
    """Test Douglas-Peucker reduces a rectangle contour to 4 points."""
    edges = detect_edges(synthetic_rect_image)
    contours = find_and_sort_contours(edges)
    approx = approximate_polygon(contours[0], epsilon_ratio=0.02)
    assert len(approx) == 4


def test_is_valid_document_quadrilateral():
    """Test geometric validation logic."""
    valid_rect = np.array([
        [[50, 50]],
        [[250, 50]],
        [[250, 250]],
        [[50, 250]]
    ], dtype=np.int32)
    is_valid, msg = is_valid_document_quadrilateral(valid_rect, image_shape=(300, 300))
    assert is_valid

    tiny_rect = np.array([
        [[10, 10]],
        [[20, 10]],
        [[20, 20]],
        [[10, 20]]
    ], dtype=np.int32)
    is_valid, msg = is_valid_document_quadrilateral(tiny_rect, image_shape=(500, 500))
    assert not is_valid
    assert "below minimum threshold" in msg

    concave = np.array([
        [[10, 10]],
        [[200, 200]],
        [[200, 10]],
        [[10, 200]]
    ], dtype=np.int32)
    is_valid, msg = is_valid_document_quadrilateral(concave, image_shape=(300, 300))
    assert not is_valid


def test_compute_quadrilateral_angles():
    """Test calculation of 4 internal angles of a rectangle (should be 90 degrees each)."""
    rect_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    angles = compute_quadrilateral_angles(rect_pts)
    assert len(angles) == 4
    for a in angles:
        assert pytest.approx(a, abs=1.0) == 90.0


def test_find_document_corners_on_sample(sample_document_path):
    """Test corner detection on the sample document photograph."""
    if not sample_document_path.exists():
        pytest.skip("Sample document image not generated.")

    img = cv2.imread(str(sample_document_path))
    corners, success, debug = find_document_corners(img)

    assert success is True
    assert corners.shape == (4, 2)
    assert debug["contours_found"] > 0
    assert "Document successfully detected" in debug["message"]


def test_find_document_corners_fallback_on_blank():
    """Test graceful fallback when no document exists in the image."""
    blank_img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    corners, success, debug = find_document_corners(blank_img)

    assert success is False
    assert corners.shape == (4, 2)
    fallback = get_default_fallback_corners((400, 400), margin_ratio=0.05)
    assert np.allclose(corners, fallback)


def test_draw_document_boundary():
    """Test drawing boundary returns a modified copy without mutating source."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img_orig = img.copy()
    corners = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], dtype=np.float32)

    annotated = draw_document_boundary(img, corners)
    assert annotated.shape == img.shape
    assert not np.array_equal(annotated, img)
    assert np.array_equal(img, img_orig)


# =========================================================================
# Phase 8: Robustness & Edge Case Tests
# =========================================================================

def test_edge_case_very_small_image():
    """Test detection on unusually small image without crashing."""
    small_img = np.zeros((60, 80, 3), dtype=np.uint8)
    cv2.rectangle(small_img, (10, 10), (70, 50), (255, 255, 255), -1)

    corners, success, debug = find_document_corners(small_img, working_height=50)
    assert corners.shape == (4, 2)
    assert isinstance(success, bool)


def test_edge_case_low_contrast_image():
    """Test detection on an extremely low contrast image."""
    # Gray values 120 vs 125
    low_contrast = np.ones((400, 400, 3), dtype=np.uint8) * 120
    cv2.rectangle(low_contrast, (50, 50), (350, 350), (125, 125, 125), -1)

    corners, success, debug = find_document_corners(low_contrast)
    assert corners.shape == (4, 2)
    # The application gracefully returns either detection or safe fallback
    assert len(corners) == 4


def test_edge_case_multiple_rectangular_objects():
    """Test that the largest prominent document quadrilateral is prioritized over small items."""
    scene = np.zeros((600, 800, 3), dtype=np.uint8)
    # 1. Tiny sticky note (30x40)
    cv2.rectangle(scene, (50, 50), (80, 90), (255, 255, 255), -1)
    # 2. Medium business card (80x120)
    cv2.rectangle(scene, (100, 400), (180, 520), (255, 255, 255), -1)
    # 3. Main physical document (350x500)
    cv2.rectangle(scene, (250, 50), (600, 550), (255, 255, 255), -1)

    corners, success, debug = find_document_corners(scene)
    assert success is True
    # Verify the detected quad corresponds to the large document, not the tiny note
    area = cv2.contourArea(corners.astype(np.int32))
    assert area > 100000  # 350 * 500 = 175000
