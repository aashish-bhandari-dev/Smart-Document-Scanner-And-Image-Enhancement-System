"""Document detection package for edges, contours, and 4-corner quad identification."""

from src.detection.edge_detection import detect_edges, auto_canny, close_edges
from src.detection.contour_detection import find_and_sort_contours, approximate_polygon
from src.detection.document_detection import (
    compute_quadrilateral_angles,
    is_valid_document_quadrilateral,
    get_default_fallback_corners,
    find_document_corners,
    draw_document_boundary,
)

__all__ = [
    "detect_edges",
    "auto_canny",
    "close_edges",
    "find_and_sort_contours",
    "approximate_polygon",
    "compute_quadrilateral_angles",
    "is_valid_document_quadrilateral",
    "get_default_fallback_corners",
    "find_document_corners",
    "draw_document_boundary",
]
