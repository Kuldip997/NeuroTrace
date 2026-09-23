"""
eye_regions.py

Responsibility: Eye landmark index definitions AND eye region extraction
(bounding box computation + cropping). No EAR math, no blink logic,
no gaze math — those stay in ear.py, blink_detector.py, gaze_estimator.py.

Index reference: MediaPipe Face Mesh topology (478-point model, iris included).
"Right eye" / "Left eye" refer to the SUBJECT's right/left (i.e. right
eye = appears on the LEFT side of a mirrored webcam frame).
"""

from typing import List, Tuple
import numpy as np

# 6-point sets used for Eye Aspect Ratio (EAR) calculation.
# Order: [outer_corner, upper_1, upper_2, inner_corner, lower_1, lower_2]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

EYE_LANDMARKS = {
    "right": RIGHT_EYE_EAR_IDX,
    "left": LEFT_EYE_EAR_IDX,
}

# Iris landmark indices — only present because our model includes the
# iris-refined face mesh (indices 468-477 are always output by the
# FaceLandmarker model bundle we downloaded, regardless of blendshapes).
RIGHT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_RING_IDX = [469, 470, 471, 472]

LEFT_IRIS_CENTER_IDX = 473
LEFT_IRIS_RING_IDX = [474, 475, 476, 477]

# Extra padding (as a fraction of eye width) added around the tight
# landmark bounding box for crops.
EYE_CROP_PADDING_RATIO = 0.4


def get_eye_bounding_box(
    landmarks,
    indices: List[int],
    frame_width: int,
    frame_height: int,
    padding_ratio: float = EYE_CROP_PADDING_RATIO,
) -> Tuple[int, int, int, int]:
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]

    x_min_norm, x_max_norm = min(xs), max(xs)
    y_min_norm, y_max_norm = min(ys), max(ys)

    eye_width_norm = x_max_norm - x_min_norm
    pad_x = eye_width_norm * padding_ratio
    pad_y = eye_width_norm * padding_ratio

    x_min = max(0, int((x_min_norm - pad_x) * frame_width))
    x_max = min(frame_width, int((x_max_norm + pad_x) * frame_width))
    y_min = max(0, int((y_min_norm - pad_y) * frame_height))
    y_max = min(frame_height, int((y_max_norm + pad_y) * frame_height))

    return x_min, y_min, x_max, y_max


def crop_eye(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        return np.array([])
    return frame[y_min:y_max, x_min:x_max]


def extract_both_eyes(landmarks, frame: np.ndarray) -> dict:
    h, w = frame.shape[:2]
    result = {}
    for eye_name, indices in EYE_LANDMARKS.items():
        bbox = get_eye_bounding_box(landmarks, indices, w, h)
        crop = crop_eye(frame, bbox)
        result[eye_name] = {"bbox": bbox, "crop": crop}
    return result