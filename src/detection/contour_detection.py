"""Contour extraction, sorting, and Douglas-Peucker polygon approximation."""

from typing import List, Tuple
import cv2
import numpy as np

from src.utils.validators import validate_image_array


def find_and_sort_contours(
    edge_map: np.ndarray,
    mode: int = cv2.RETR_LIST,
    method: int = cv2.CHAIN_APPROX_SIMPLE
) -> List[np.ndarray]:
    """Extract all contours from a binary edge map and sort them by area descending.
    
    Args:
        edge_map: 2D binary edge image.
        mode: OpenCV contour retrieval mode (e.g. cv2.RETR_LIST, cv2.RETR_EXTERNAL).
        method: OpenCV contour approximation method (e.g. cv2.CHAIN_APPROX_SIMPLE).
        
    Returns:
        List of NumPy contour arrays sorted from largest area to smallest area.
    """
    validate_image_array(edge_map)
    
    # cv2.findContours modifies image in older OpenCV, pass a copy
    contours, _ = cv2.findContours(edge_map.copy(), mode, method)
    if not contours:
        return []

    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return sorted_contours


def approximate_polygon(
    contour: np.ndarray,
    epsilon_ratio: float = 0.02
) -> np.ndarray:
    """Approximate a contour into a simplified polygon using Douglas-Peucker algorithm.
    
    The algorithm recursively finds the point furthest from the line connecting
    the endpoints. If the distance exceeds epsilon = epsilon_ratio * perimeter,
    the curve is subdivided.
    
    Args:
        contour: Input contour as NumPy array.
        epsilon_ratio: Precision parameter specifying approximation accuracy as a
                      fraction of the contour perimeter (typical range 0.01 - 0.05).
                      
    Returns:
        Approximated polygon vertices as a NumPy array with shape (N, 1, 2).
    """
    if contour is None or len(contour) < 3:
        return np.array([], dtype=np.int32)

    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter <= 0:
        return np.array([], dtype=np.int32)

    epsilon = epsilon_ratio * perimeter
    approximated = cv2.approxPolyDP(contour, epsilon, closed=True)
    return approximated
