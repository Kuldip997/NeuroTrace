"""
eye_capture_model.py

Responsibility: The PUBLIC interface for this entire module. This is
the only file another developer/team should need to import.

Usage:
    model = EyeCaptureModel()
    features = model.process_frame(frame, timestamp)   # per frame
    ...
    summary = model.get_session_features()             # at session end
    model.reset()                                        # to start a new session
    model.close()                                        # when fully done
"""

from face_detector import FaceDetector
from eye_regions import RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX
from ear import calculate_both_ears
from blink_detector import BlinkDetector
from gaze_estimator import estimate_eye_movement
from feature_aggregator import SessionAggregator
from quality import validate_frame
from schema import FrameFeatures, SessionFeatures

import numpy as np
import os

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")


class EyeCaptureModel:
    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self._detector = FaceDetector(model_path=model_path)
        self._blink_detector = BlinkDetector()
        self._aggregator = SessionAggregator()

        self._total_frame_count = 0
        self._valid_frame_count = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameFeatures:
        """
        Processes a single BGR frame and returns its extracted features.
        Also feeds valid frames into session-level aggregation internally.

        Args:
            frame: BGR frame, e.g. from cv2.VideoCapture.
            timestamp: seconds (e.g. time.time()), must be increasing
                across calls within a session.

        Returns:
            A FrameFeatures record describing this frame.
        """
        self._total_frame_count += 1
        landmarks = self._detector.detect(frame)
        h, w = frame.shape[:2]

        if landmarks is None:
            return FrameFeatures(
                timestamp=timestamp,
                face_detected=False,
                quality_reasons=["no_face_detected"],
            )

        ears = calculate_both_ears(landmarks, RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX)
        mean_ear = ears["mean_ear"]
        movement = estimate_eye_movement(landmarks)

        bbox = self._detector.get_bounding_box(landmarks, w, h)
        quality = validate_frame(landmarks, frame, bbox)

        blink_event = self._blink_detector.update(mean_ear, timestamp)

        # Only feed quality-valid frames into session statistics —
        # this is what keeps blink rate / EAR stats accurate (see
        # PROJECT_TRACKER.md Known Issues / Stage 9 rationale).
        if quality.is_valid:
            self._aggregator.update(mean_ear, timestamp, movement)
            self._valid_frame_count += 1

        return FrameFeatures(
            timestamp=timestamp,
            face_detected=True,
            left_ear=ears["left_ear"],
            right_ear=ears["right_ear"],
            mean_ear=mean_ear,
            blink_detected=blink_event is not None,
            eye_movement_horizontal_label=movement["right_eye"]["horizontal_label"],
            eye_movement_horizontal_ratio=movement["right_eye"]["horizontal_ratio"],
            eye_movement_vertical_label=movement["right_eye"]["vertical_label"],
            eye_movement_vertical_ratio=movement["right_eye"]["vertical_ratio"],
            quality_valid=quality.is_valid,
            quality_score=quality.quality_score,
            quality_reasons=quality.reasons,
        )

    def get_session_features(self) -> SessionFeatures:
        """
        Returns the accumulated session-level summary statistics,
        based on all quality-valid frames processed so far via
        process_frame(). Safe to call mid-session or at the end.
        """
        agg_stats = self._aggregator.get_session_features(self._blink_detector)

        valid_pct = 0.0
        if self._total_frame_count > 0:
            valid_pct = round(100.0 * self._valid_frame_count / self._total_frame_count, 1)

        return SessionFeatures(
            session_duration_sec=agg_stats.get("session_duration_sec", 0.0),
            total_frames=self._total_frame_count,
            valid_frames=self._valid_frame_count,
            valid_frames_pct=valid_pct,
            mean_ear=agg_stats.get("mean_ear", 0.0),
            ear_variance=agg_stats.get("ear_variance", 0.0),
            blink_count=agg_stats.get("blink_count", 0),
            blink_rate_per_min=agg_stats.get("blink_rate_per_min", 0.0),
            mean_blink_duration_ms=agg_stats.get("mean_blink_duration_ms", 0.0),
            eye_movement_horizontal_distribution_pct=agg_stats.get("eye_movement_horizontal_distribution_pct", {}),
            eye_movement_vertical_distribution_pct=agg_stats.get("eye_movement_vertical_distribution_pct", {}),
        )

    def reset(self):
        """
        Clears all session state (blink history, EAR history, movement
        counts) to start a fresh session, WITHOUT reloading the
        MediaPipe model — that's the expensive part, so this stays fast.
        """
        self._blink_detector.reset()
        self._aggregator.reset()
        self._total_frame_count = 0
        self._valid_frame_count = 0

    def close(self):
        """
        Releases the underlying MediaPipe model. Call this once, when
        completely done with the model (not between sessions — use
        reset() for that).
        """
        self._detector.close()