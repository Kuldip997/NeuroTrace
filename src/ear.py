"""
ear.py

Responsibility: ONLY the Eye Aspect Ratio (EAR) math. Pure functions,
no camera, no drawing, no state. Takes landmark points in, returns
numbers out — this makes it trivially unit-testable.
"""

import math
from typing import List


def _euclidean_distance(p1, p2) -> float:
    """
    Distance between two landmark points using their normalized (x, y)
    coordinates. Since both points come from the same frame, the
    normalization scale cancels out in the EAR ratio, so we don't need
    to convert to pixel coordinates first.
    """
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calculate_ear(eye_points: List) -> float:
    """
    Calculates Eye Aspect Ratio for one eye.

    Args:
        eye_points: list of 6 landmark points in the fixed order
            [outer_corner, upper_1, upper_2, inner_corner, lower_1, lower_2]
            — this order MUST match RIGHT_EYE_EAR_IDX / LEFT_EYE_EAR_IDX
            in eye_regions.py.

    Returns:
        EAR value as a float. Roughly 0.25-0.35 = open, <0.15 = closing/closed.
    """
    if len(eye_points) != 6:
        raise ValueError(f"calculate_ear expects exactly 6 points, got {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = _euclidean_distance(p2, p6)
    vertical_2 = _euclidean_distance(p3, p5)
    horizontal = _euclidean_distance(p1, p4)

    if horizontal == 0:
        # Degenerate case (shouldn't normally happen with real landmarks) —
        # avoid a divide-by-zero crash, return 0 (treated as "fully closed/invalid").
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def calculate_both_ears(landmarks, right_eye_indices: List[int], left_eye_indices: List[int]) -> dict:
    """
    Convenience function: computes right EAR, left EAR, and mean EAR
    from the full 478-point landmark list plus the eye index lists.

    Returns:
        {"right_ear": float, "left_ear": float, "mean_ear": float}
    """
    right_points = [landmarks[i] for i in right_eye_indices]
    left_points = [landmarks[i] for i in left_eye_indices]

    right_ear = calculate_ear(right_points)
    left_ear = calculate_ear(left_points)
    mean_ear = (right_ear + left_ear) / 2.0

    return {"right_ear": right_ear, "left_ear": left_ear, "mean_ear": mean_ear}