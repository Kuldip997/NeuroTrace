"""
quality.py

Responsibility: ONLY per-frame quality validation. Determines whether
a frame's data should be trusted for feature extraction, and why not
if it shouldn't be. No camera, no EAR/blink logic — just checks.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List


# --- Configuration (tune based on your camera/lighting setup) ---
YAW_RATIO_MIN = 0.35   # nose-position ratio outside this range = turned too far
YAW_RATIO_MAX = 0.65
PITCH_RATIO_MIN = 0.35
PITCH_RATIO_MAX = 0.65

MIN_BRIGHTNESS = 40.0        # mean pixel value (0-255) below this = too dark
MIN_SHARPNESS = 50.0         # Laplacian variance below this = too blurry
MIN_FACE_WIDTH_RATIO = 0.15  # face bbox width / frame width below this = too far


@dataclass
class QualityResult:
    is_valid: bool
    quality_score: float          # 0.0 (unusable) to 1.0 (ideal)
    reasons: List[str] = field(default_factory=list)  # why invalid, if not valid


def estimate_yaw_ratio(landmarks) -> float:
    """
    Coarse yaw proxy: nose tip (idx 1) position between right face edge
    (idx 234) and left face edge (idx 454). ~0.5 = facing camera straight.
    NOT true 3D head pose — a lightweight 2D geometric approximation.
    """
    nose = landmarks[1]
    right_edge = landmarks[234]
    left_edge = landmarks[454]

    face_width = left_edge.x - right_edge.x
    if face_width == 0:
        return 0.5
    return (nose.x - right_edge.x) / face_width


def estimate_pitch_ratio(landmarks) -> float:
    """
    Coarse pitch proxy: nose tip (idx 1) position between forehead
    (idx 10) and chin (idx 152). ~0.45-0.5 = facing camera straight
    (nose sits slightly above the forehead-chin midpoint naturally).
    """
    nose = landmarks[1]
    forehead = landmarks[10]
    chin = landmarks[152]

    face_height = chin.y - forehead.y
    if face_height == 0:
        return 0.5
    return (nose.y - forehead.y) / face_height


def compute_brightness(gray_region: np.ndarray) -> float:
    """Mean pixel intensity of a grayscale region. Higher = brighter."""
    if gray_region.size == 0:
        return 0.0
    return float(np.mean(gray_region))


def compute_sharpness(gray_region: np.ndarray) -> float:
    """
    Variance of the Laplacian — a standard, cheap blur metric.
    Sharp edges produce high-variance Laplacian response; blur smooths
    edges out, lowering the variance.
    """
    if gray_region.size == 0:
        return 0.0
    laplacian = cv2.Laplacian(gray_region, cv2.CV_64F)
    return float(laplacian.var())


def validate_frame(landmarks, frame: np.ndarray, face_bbox: tuple) -> QualityResult:
    """
    Runs all quality checks on one frame and returns a combined result.

    Args:
        landmarks: detected face landmarks (None if no face detected).
        frame: the full BGR frame.
        face_bbox: (x_min, y_min, x_max, y_max) of the face, from FaceDetector.

    Returns:
        QualityResult with is_valid, quality_score, and reasons for failure.
    """
    reasons = []

    if landmarks is None:
        return QualityResult(is_valid=False, quality_score=0.0, reasons=["no_face_detected"])

    x_min, y_min, x_max, y_max = face_bbox
    frame_h, frame_w = frame.shape[:2]

    # --- Face size check ---
    face_width_ratio = (x_max - x_min) / frame_w if frame_w > 0 else 0.0
    if face_width_ratio < MIN_FACE_WIDTH_RATIO:
        reasons.append("face_too_small_or_far")

    # --- Head pose checks ---
    yaw_ratio = estimate_yaw_ratio(landmarks)
    if not (YAW_RATIO_MIN <= yaw_ratio <= YAW_RATIO_MAX):
        reasons.append("head_yaw_too_extreme")

    pitch_ratio = estimate_pitch_ratio(landmarks)
    if not (PITCH_RATIO_MIN <= pitch_ratio <= PITCH_RATIO_MAX):
        reasons.append("head_pitch_too_extreme")

    # --- Lighting and blur checks (on the face region only) ---
    face_crop = frame[max(0, y_min):y_max, max(0, x_min):x_max]
    brightness = 0.0
    sharpness = 0.0
    if face_crop.size > 0:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = compute_brightness(gray)
        sharpness = compute_sharpness(gray)

        if brightness < MIN_BRIGHTNESS:
            reasons.append("poor_lighting")
        if sharpness < MIN_SHARPNESS:
            reasons.append("motion_blur")
    else:
        reasons.append("face_crop_invalid")

    is_valid = len(reasons) == 0

    # Simple composite score: 1.0 minus a penalty per failed check,
    # floored at 0. Not a rigorous probability — a rough usability signal.
    quality_score = max(0.0, 1.0 - 0.2 * len(reasons))

    return QualityResult(is_valid=is_valid, quality_score=quality_score, reasons=reasons)