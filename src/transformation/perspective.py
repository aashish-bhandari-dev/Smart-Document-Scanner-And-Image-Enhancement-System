"""Perspective transformation and orthographic document rectification."""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.validators import validate_image_array


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four 2D points into consistent canonical order: [TL, TR, BR, BL].
    
    The ordering algorithm uses mathematical sum and difference properties:
    - Top-Left (TL): Has the smallest (x + y) coordinate sum.
    - Bottom-Right (BR): Has the largest (x + y) coordinate sum.
    - Top-Right (TR): Has the smallest (y - x) coordinate difference.
    - Bottom-Left (BL): Has the largest (y - x) coordinate difference.
    
    Args:
        pts: NumPy array of shape (4, 2) or (4, 1, 2) containing corner coordinates.
        
    Returns:
        NumPy float32 array of shape (4, 2) ordered:
        [0]: Top-Left (TL)
        [1]: Top-Right (TR)
        [2]: Bottom-Right (BR)
        [3]: Bottom-Left (BL)
        
    Raises:
        ValueError: If input does not contain exactly 4 points.
    """
    if pts is None:
        raise ValueError("Corner points array cannot be None.")

    pts = np.asarray(pts, dtype=np.float32).reshape((-1, 2))
    if pts.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got {pts.shape[0]}.")

    ordered = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: (x + y)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # Top-Left: smallest sum
    ordered[2] = pts[np.argmax(s)]  # Bottom-Right: largest sum

    # Difference of coordinates: (y - x)
    diff = np.diff(pts, axis=1).flatten()  # y - x
    ordered[1] = pts[np.argmin(diff)]  # Top-Right: smallest diff (x is large, y is small)
    ordered[3] = pts[np.argmax(diff)]  # Bottom-Left: largest diff (x is small, y is large)

    return ordered


def compute_destination_dimensions(ordered_pts: np.ndarray) -> Tuple[int, int]:
    """Calculate target width and height of the perspective-corrected document.
    
    Determines dimensions based on the maximum Euclidean distances between opposite edges:
    - Target width: max(distance(TL, TR), distance(BL, BR))
    - Target height: max(distance(TL, BL), distance(TR, BR))
    
    This preserves physical document proportions and prevents anisotropic stretching.
    
    Args:
        ordered_pts: (4, 2) float32 array in canonical order [TL, TR, BR, BL].
        
    Returns:
        Tuple of (width: int, height: int).
    """
    tl, tr, br, bl = ordered_pts

    # Compute Euclidean distance of top and bottom horizontal edges
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(round(width_top), round(width_bottom)))

    # Compute Euclidean distance of left and right vertical edges
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(round(height_left), round(height_right)))

    # Ensure minimum 1x1 pixel dimension
    max_width = max(1, max_width)
    max_height = max(1, max_height)

    return max_width, max_height


def warp_perspective(
    image: np.ndarray,
    pts: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """Correct perspective distortion of a document using projective homography.
    
    Computes a 3x3 projective transformation matrix M from ordered document corners
    to an orthogonal top-down rectangle, and warps the image using cv2.warpPerspective.
    
    Args:
        image: Source image array (BGR, RGB, or Grayscale).
        pts: Four corner coordinates of the document boundary.
        output_size: Optional explicit (width, height) tuple. If None, optimal dimensions
                     are computed automatically from edge lengths.
        interpolation: OpenCV interpolation algorithm. Default is cv2.INTER_CUBIC for
                       sharp, high-fidelity text preservation.
                       
    Returns:
        Perspective-corrected rectified image array with shape (H, W) or (H, W, C).
        
    Raises:
        ImageValidationError: If image is invalid.
        ValueError: If corner points are invalid.
    """
    validate_image_array(image)
    ordered_src = order_points(pts)

    if output_size is not None:
        target_w, target_h = int(output_size[0]), int(output_size[1])
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Output size dimensions must be positive, got {output_size}.")
    else:
        target_w, target_h = compute_destination_dimensions(ordered_src)

    # Destination coordinates for canonical flat rectangular view
    dst_pts = np.array([
        [0, 0],                         # Top-Left
        [target_w - 1, 0],              # Top-Right
        [target_w - 1, target_h - 1],   # Bottom-Right
        [0, target_h - 1]               # Bottom-Left
    ], dtype=np.float32)

    # Calculate 3x3 perspective homography matrix
    matrix = cv2.getPerspectiveTransform(ordered_src, dst_pts)

    # Apply projective warp
    warped = cv2.warpPerspective(
        image,
        matrix,
        (target_w, target_h),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped
