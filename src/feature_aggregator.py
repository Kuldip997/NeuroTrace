"""
feature_aggregator.py

Responsibility: ONLY session-level statistics. Consumes a stream of
per-frame values (mean EAR, timestamp, eye-movement labels) fed to it
one frame at a time, and produces summary statistics on demand.

No camera, no landmarks, no detection logic — purely accumulation + math.
This is what makes per-frame data actually useful to a downstream
consumer: raw frames aren't the deliverable, session statistics are.
"""

import statistics
from typing import Optional, List


class SessionAggregator:
    """
    Accumulates per-frame features over a session and computes
    summary statistics: blink stats, EAR stats, eye-movement distribution.
    """

    def __init__(self):
        self._session_start_time: Optional[float] = None
        self._last_timestamp: Optional[float] = None

        self._ear_values: List[float] = []

        # Counts per label, for computing distributions later.
        self._horizontal_label_counts = {"outward": 0, "inward": 0, "center": 0}
        self._vertical_label_counts = {"up": 0, "down": 0, "center": 0}
        self._valid_movement_frames = 0

    def update(self, mean_ear: float, timestamp: float,
               movement: Optional[dict] = None) -> None:
        """
        Feeds one frame's data into the aggregator.

        Args:
            mean_ear: this frame's mean EAR value.
            timestamp: this frame's timestamp (seconds, e.g. time.time()).
            movement: optional dict from gaze_estimator.estimate_eye_movement().
                      If provided, its labels are tallied for distribution stats.
        """
        if self._session_start_time is None:
            self._session_start_time = timestamp
        self._last_timestamp = timestamp

        self._ear_values.append(mean_ear)

        if movement is not None:
            # Use the right eye's labels as the representative signal —
            # both eyes were shown to move together in Stage 7 testing,
            # so one eye is sufficient for a distribution estimate.
            h_label = movement["right_eye"]["horizontal_label"]
            v_label = movement["right_eye"]["vertical_label"]
            self._horizontal_label_counts[h_label] += 1
            self._vertical_label_counts[v_label] += 1
            self._valid_movement_frames += 1

    def _label_distribution(self, counts: dict) -> dict:
        """Converts raw counts into percentages, safely handling zero frames."""
        total = sum(counts.values())
        if total == 0:
            return {label: 0.0 for label in counts}
        return {label: round(100.0 * count / total, 1) for label, count in counts.items()}

    def get_session_features(self, blink_detector) -> dict:
        """
        Computes the full session summary. Takes the BlinkDetector instance
        directly (rather than duplicating blink storage here) so blink
        stats always reflect the single source of truth.

        Returns a dict — this shape is a working draft; Stage 10 will
        formalize it into the final output schema.
        """
        if self._session_start_time is None or self._last_timestamp is None:
            # No frames processed yet.
            return {"error": "no_data", "message": "No frames have been processed yet."}

        session_duration_sec = self._last_timestamp - self._session_start_time

        # --- EAR stats ---
        mean_ear = statistics.mean(self._ear_values) if self._ear_values else 0.0
        ear_variance = statistics.pvariance(self._ear_values) if len(self._ear_values) > 1 else 0.0

        # --- Blink stats ---
        blink_count = blink_detector.total_blinks
        blink_durations_ms = [e.duration_sec * 1000 for e in blink_detector.blink_events]
        mean_blink_duration_ms = statistics.mean(blink_durations_ms) if blink_durations_ms else 0.0

        blink_rate_per_min = 0.0
        if session_duration_sec > 0:
            blink_rate_per_min = blink_count / (session_duration_sec / 60.0)

        # --- Eye movement distribution ---
        horizontal_distribution = self._label_distribution(self._horizontal_label_counts)
        vertical_distribution = self._label_distribution(self._vertical_label_counts)

        return {
            "session_duration_sec": round(session_duration_sec, 2),
            "frame_count": len(self._ear_values),

            "mean_ear": round(mean_ear, 4),
            "ear_variance": round(ear_variance, 6),

            "blink_count": blink_count,
            "blink_rate_per_min": round(blink_rate_per_min, 2),
            "mean_blink_duration_ms": round(mean_blink_duration_ms, 1),

            "eye_movement_horizontal_distribution_pct": horizontal_distribution,
            "eye_movement_vertical_distribution_pct": vertical_distribution,
            "eye_movement_vertical_confidence": "low",  # per Known Issue in tracker
        }

    def reset(self):
        """Resets all accumulated state — used when starting a new session."""
        self._session_start_time = None
        self._last_timestamp = None
        self._ear_values = []
        self._horizontal_label_counts = {"outward": 0, "inward": 0, "center": 0}
        self._vertical_label_counts = {"up": 0, "down": 0, "center": 0}
        self._valid_movement_frames = 0