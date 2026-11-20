"""Four-corner document boundary detection, validation, and visualization."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.preprocessing.filters import apply_gaussian_blur, to_grayscale
from src.preprocessing.resize import resize_image
from src.detection.edge_detection import auto_canny, close_edges, detect_edges
from src.detection.contour_detection import approximate_polygon, find_and_sort_contours
from src.utils.validators import validate_image_array


def compute_quadrilateral_angles(pts: np.ndarray) -> List[float]:
    """Compute the 4 interior angles (in degrees) of a quadrilateral.
    
    Args:
        pts: (4, 2) array of corner coordinates.
        
    Returns:
        List of 4 interior angles in degrees.
    """
    angles = []
    for i in range(4):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[(i + 1) % 4]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            angles.append(0.0)
            continue

        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        angles.append(float(angle))

    return angles


def is_valid_document_quadrilateral(
    approx_poly: np.ndarray,
    image_shape: Tuple[int, int],
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.98,
    min_angle: float = 45.0,
    max_angle: float = 135.0
) -> Tuple[bool, str]:
    """Validate whether an approximated polygon represents a plausible document.
    
    Checks:
    1. Exactly 4 vertices.
    2. Strictly convex quadrilateral.
    3. Area between min_area_ratio and max_area_ratio of total image area.
    4. Interior angles between min_angle and max_angle.
    
    Args:
        approx_poly: Approximated contour array of shape (4, 1, 2) or (4, 2).
        image_shape: Tuple of (height, width).
        min_area_ratio: Minimum fraction of total image area (filters tiny objects).
        max_area_ratio: Maximum fraction of total image area (filters camera boundary).
        min_angle: Minimum allowed corner angle in degrees.
        max_angle: Maximum allowed corner angle in degrees.
        
    Returns:
        Tuple of (is_valid: bool, reason: str).
    """
    if approx_poly is None or len(approx_poly) != 4:
        return False, f"Polygon does not have 4 vertices (has {len(approx_poly) if approx_poly is not None else 0})."

    # Ensure shape (4, 2)
    pts = approx_poly.reshape(4, 2).astype(np.float32)

    # 1. Convexity check
    if not cv2.isContourConvex(pts.astype(np.int32)):
        return False, "Quadrilateral is concave or self-intersecting."

    # 2. Area ratio check
    h, w = image_shape[:2]
    total_area = float(h * w)
    quad_area = cv2.contourArea(pts.astype(np.int32))
    area_ratio = quad_area / total_area

    if area_ratio < min_area_ratio:
        return False, f"Area ratio {area_ratio:.3f} is below minimum threshold {min_area_ratio}."

    if area_ratio > max_area_ratio:
        return False, f"Area ratio {area_ratio:.3f} exceeds maximum threshold {max_area_ratio}."

    # 3. Angle validation
    angles = compute_quadrilateral_angles(pts)
    for angle in angles:
        if angle < min_angle or angle > max_angle:
            return False, f"Internal angle {angle:.1f}° outside valid range [{min_angle}°, {max_angle}°]."

    return True, "Valid document quadrilateral."


def get_default_fallback_corners(image_shape: Tuple[int, int], margin_ratio: float = 0.05) -> np.ndarray:
    """Generate default corner coordinates inset from the image boundaries.
    
    Used when automatic detection fails so the application never crashes.
    
    Args:
        image_shape: (height, width).
        margin_ratio: Margin fraction from edge (default 5%).
        
    Returns:
        (4, 2) NumPy array of float32 coordinates ordered [TL, TR, BR, BL].
    """
    h, w = image_shape[:2]
    mx = w * margin_ratio
    my = h * margin_ratio

    return np.array([
        [mx, my],            # Top-Left
        [w - mx, my],        # Top-Right
        [w - mx, h - my],    # Bottom-Right
        [mx, h - my]         # Bottom-Left
    ], dtype=np.float32)


def find_document_corners(
    image: np.ndarray,
    working_height: int = 800,
    low_canny: Optional[int] = None,
    high_canny: Optional[int] = None,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.98
) -> Tuple[np.ndarray, bool, Dict[str, any]]:
    """Detect the 4 corners of a physical document in an image.
    
    Pipeline:
    1. Downscales image preserving aspect ratio and computes scale ratio.
    2. Converts downscaled image to grayscale and applies Gaussian blur.
    3. Runs Canny edge detector (adaptive or custom) and morphological closing.
    4. Extracts contours and sorts by area.
    5. Iterates through largest candidate contours, applies Douglas-Peucker approximation,
       and validates candidate 4-corner polygons.
    6. If found, projects corners back to original image dimensions by multiplying by scale ratio.
    7. If not found, falls back gracefully to an inset quadrilateral.
    
    Args:
        image: Original BGR or Grayscale image array.
        working_height: Height to resize to for efficient processing.
        low_canny: Optional explicit lower Canny threshold.
        high_canny: Optional explicit upper Canny threshold.
        min_area_ratio: Minimum area fraction of working image.
        max_area_ratio: Maximum area fraction of working image.
        
    Returns:
        Tuple of:
        - corners: (4, 2) float32 array in original image coordinates.
        - success: True if a valid 4-corner document was detected, False if fallback was used.
        - debug_info: Dictionary containing intermediate edge_map, closed_map, ratio, message.
    """
    validate_image_array(image)
    orig_h, orig_w = image.shape[:2]

    # 1. Resize for fast processing while recording scale ratio
    resized, ratio = resize_image(image, height=working_height)
    res_h, res_w = resized.shape[:2]

    # 2. Grayscale & Gaussian blur
    gray = to_grayscale(resized)
    blurred = apply_gaussian_blur(gray, kernel_size=(5, 5))

    # 3. Edge detection
    if low_canny is not None and high_canny is not None:
        edges = detect_edges(blurred, low_threshold=low_canny, high_threshold=high_canny)
    else:
        edges = auto_canny(blurred)

    # Morphological closing to seal small edge breaks
    closed_edges = close_edges(edges, kernel_size=(5, 5))

    # 4. Extract and sort contours
    contours = find_and_sort_contours(closed_edges)

    # 5. Search for a valid 4-point quadrilateral among the largest contours
    detected_corners_resized: Optional[np.ndarray] = None
    detection_reason = "No contours extracted."

    # Search among top 15 largest contours
    for cnt in contours[:15]:
        # Test varying epsilon ratios (0.015 to 0.04) for robustness
        for eps_ratio in [0.02, 0.03, 0.015, 0.04]:
            approx = approximate_polygon(cnt, epsilon_ratio=eps_ratio)
            if len(approx) == 4:
                is_valid, reason = is_valid_document_quadrilateral(
                    approx,
                    (res_h, res_w),
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio
                )
                if is_valid:
                    detected_corners_resized = approx.reshape(4, 2).astype(np.float32)
                    detection_reason = "Document successfully detected."
                    break
        if detected_corners_resized is not None:
            break

    # 6. Scale coordinates or fallback
    if detected_corners_resized is not None:
        # Scale back to full-resolution coordinates
        corners_orig = detected_corners_resized * ratio
        success = True
    else:
        # Graceful fallback: 5% inset of full-resolution image
        corners_orig = get_default_fallback_corners((orig_h, orig_w), margin_ratio=0.05)
        success = False
        detection_reason = (
            "Automatic boundary detection could not isolate a clear 4-corner document. "
            "Using default fallback boundary."
        )

    debug_info = {
        "success": success,
        "message": detection_reason,
        "ratio": ratio,
        "resized_dimensions": (res_w, res_h),
        "edge_map": edges,
        "closed_edges": closed_edges,
        "contours_found": len(contours),
    }

    return corners_orig, success, debug_info


def draw_document_boundary(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    draw_corners: bool = True
) -> np.ndarray:
    """Draw the detected document boundary polygon and corner markers on an image copy.
    
    Args:
        image: Source image (BGR or RGB).
        corners: (4, 2) array of corner coordinates.
        color: Boundary line color in BGR/RGB tuple (default bright green).
        thickness: Line thickness in pixels.
        draw_corners: Whether to draw circular markers at the 4 vertices.
        
    Returns:
        New image array with boundary annotations drawn.
    """
    validate_image_array(image)
    annotated = image.copy()
    pts = np.int32(corners).reshape((-1, 1, 2))

    # Draw closed quadrilateral boundary
    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)

    # Draw corner markers
    if draw_corners:
        radius = max(4, int(max(image.shape[:2]) * 0.008))
        for idx, pt in enumerate(corners):
            x, y = int(round(pt[0])), int(round(pt[1]))
            # Outer circle
            cv2.circle(annotated, (x, y), radius, (0, 0, 255), -1)  # Red marker
            # Inner circle
            cv2.circle(annotated, (x, y), max(2, radius // 2), (255, 255, 255), -1)

    return annotated
