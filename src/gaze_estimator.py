"""
gaze_estimator.py

Responsibility: ONLY coarse eye-movement estimation from iris landmark
position relative to eye corners.

IMPORTANT — read before using this module's output:
This is NOT screen-gaze tracking. It estimates where the iris sits
RELATIVE TO THE EYE OPENING, not where a person is looking on a screen.
It does not correct for head pose, so turning the head while keeping
eyes centered will read as "center" even though the head direction
changed. True gaze-point tracking would require head-pose correction
plus a per-user calibration step, which is out of scope for this module.
"""

from typing import Tuple

# Ratio thresholds for classifying horizontal iris position.
# Tuned to be a reasonable default — adjust while watching live values
# on your own camera/lighting setup.
HORIZONTAL_LEFT_THRESHOLD = 0.35
HORIZONTAL_RIGHT_THRESHOLD = 0.65

# Vertical thresholds — wider deadzone since vertical estimation is
# noisier than horizontal (smaller physical range of motion).
VERTICAL_UP_THRESHOLD = 0.35
VERTICAL_DOWN_THRESHOLD = 0.65


def _get_point(landmarks, idx: int):
    return landmarks[idx]


def estimate_horizontal_position(landmarks, outer_idx: int, inner_idx: int, iris_center_idx: int) -> float:
    """
    Returns a ratio in roughly [0,1] describing where the iris sits
    between the eye's outer corner (0.0) and inner corner (1.0).
    ~0.5 = centered.
    """
    outer = _get_point(landmarks, outer_idx)
    inner = _get_point(landmarks, inner_idx)
    iris = _get_point(landmarks, iris_center_idx)

    eye_width = inner.x - outer.x
    if eye_width == 0:
        return 0.5  # degenerate case, assume centered rather than crash

    ratio = (iris.x - outer.x) / eye_width
    return ratio


def estimate_vertical_position(landmarks, upper_idx: int, lower_idx: int, iris_center_idx: int) -> float:
    """
    Returns a ratio in roughly [0,1] describing where the iris sits
    between the eye's upper lid point (0.0) and lower lid point (1.0).
    ~0.5 = centered. Noisier than horizontal — treat as lower-confidence.
    """
    upper = _get_point(landmarks, upper_idx)
    lower = _get_point(landmarks, lower_idx)
    iris = _get_point(landmarks, iris_center_idx)

    eye_height = lower.y - upper.y
    if eye_height == 0:
        return 0.5

    ratio = (iris.y - upper.y) / eye_height
    return ratio


def classify_horizontal(ratio: float) -> str:
    if ratio < HORIZONTAL_LEFT_THRESHOLD:
        return "outward"   # toward outer corner
    elif ratio > HORIZONTAL_RIGHT_THRESHOLD:
        return "inward"    # toward inner corner (nose side)
    return "center"


def classify_vertical(ratio: float) -> str:
    if ratio < VERTICAL_UP_THRESHOLD:
        return "up"
    elif ratio > VERTICAL_DOWN_THRESHOLD:
        return "down"
    return "center"


def estimate_eye_movement(landmarks) -> dict:
    """
    Computes coarse horizontal/vertical iris position for both eyes.

    Returns a dict with raw ratios and classified labels, per eye,
    plus a combined estimate. This is descriptive, relative-to-head-pose
    positioning — NOT a screen coordinate.
    """
    # Right eye: outer=33, inner=133, upper~=159 (approx lid center), lower~=145
    right_h = estimate_horizontal_position(landmarks, outer_idx=33, inner_idx=133, iris_center_idx=468)
    right_v = estimate_vertical_position(landmarks, upper_idx=159, lower_idx=145, iris_center_idx=468)

    # Left eye: outer=362, inner=263, upper~=386, lower~=374
    left_h = estimate_horizontal_position(landmarks, outer_idx=362, inner_idx=263, iris_center_idx=473)
    left_v = estimate_vertical_position(landmarks, upper_idx=386, lower_idx=374, iris_center_idx=473)

    return {
        "right_eye": {
            "horizontal_ratio": right_h,
            "vertical_ratio": right_v,
            "horizontal_label": classify_horizontal(right_h),
            "vertical_label": classify_vertical(right_v),
        },
        "left_eye": {
            "horizontal_ratio": left_h,
            "vertical_ratio": left_v,
            "horizontal_label": classify_horizontal(left_h),
            "vertical_label": classify_vertical(left_v),
        },
    }