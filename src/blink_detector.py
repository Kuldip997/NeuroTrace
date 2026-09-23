"""
blink_detector.py

Responsibility: ONLY the temporal blink state machine. Consumes a stream
of (mean_ear, timestamp) pairs and emits blink events. No camera, no
landmarks, no drawing — purely stateful logic over EAR values.
"""

from dataclasses import dataclass
from typing import Optional, List


# --- Configuration (tune these if blink detection misbehaves) ---
EAR_THRESHOLD = 0.21          # below this = eye considered "closing/closed"
MIN_CONSEC_CLOSED_FRAMES = 2  # frames EAR must stay below threshold before
                               # we trust it's a real closure, not noise
MIN_BLINK_DURATION_SEC = 0.05 # shorter than this = discard as noise
MAX_BLINK_DURATION_SEC = 0.5  # longer than this = not a blink (eyes closed
                               # for another reason), discard as a blink event


@dataclass
class BlinkEvent:
    """A single completed, validated blink."""
    start_time: float
    end_time: float
    duration_sec: float


class BlinkDetector:
    """
    Tracks eye-open/closed state across frames using EAR values and
    timestamps, emitting a BlinkEvent whenever a complete, valid blink
    is detected. Keeps a full history of blink events for session-level
    statistics (used by feature_aggregator.py).
    """

    def __init__(self,
                 ear_threshold: float = EAR_THRESHOLD,
                 min_consec_closed_frames: int = MIN_CONSEC_CLOSED_FRAMES,
                 min_blink_duration: float = MIN_BLINK_DURATION_SEC,
                 max_blink_duration: float = MAX_BLINK_DURATION_SEC):
        self.ear_threshold = ear_threshold
        self.min_consec_closed_frames = min_consec_closed_frames
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration

        self._state = "OPEN"           # "OPEN" or "CLOSED"
        self._below_threshold_streak = 0
        self._closed_start_time: Optional[float] = None

        self.total_blinks = 0
        self.last_blink: Optional[BlinkEvent] = None
        self.blink_events: List[BlinkEvent] = []  # full session history

    def update(self, mean_ear: float, timestamp: float) -> Optional[BlinkEvent]:
        """
        Feeds one frame's mean EAR + timestamp into the state machine.

        Returns:
            A BlinkEvent if a valid blink just completed on this call,
            otherwise None.
        """
        is_below = mean_ear < self.ear_threshold

        if self._state == "OPEN":
            if is_below:
                self._below_threshold_streak += 1
                if self._below_threshold_streak >= self.min_consec_closed_frames:
                    self._state = "CLOSED"
                    self._closed_start_time = timestamp
            else:
                self._below_threshold_streak = 0

            return None

        elif self._state == "CLOSED":
            if not is_below:
                duration = timestamp - self._closed_start_time
                self._state = "OPEN"
                self._below_threshold_streak = 0
                self._closed_start_time = None

                if self.min_blink_duration <= duration <= self.max_blink_duration:
                    event = BlinkEvent(
                        start_time=timestamp - duration,
                        end_time=timestamp,
                        duration_sec=duration
                    )
                    self.total_blinks += 1
                    self.last_blink = event
                    self.blink_events.append(event)
                    return event
                return None

            return None

    def reset(self):
        """Resets all state — used when starting a new session."""
        self._state = "OPEN"
        self._below_threshold_streak = 0
        self._closed_start_time = None
        self.total_blinks = 0
        self.last_blink = None
        self.blink_events = []