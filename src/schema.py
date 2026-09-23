"""
schema.py

Responsibility: ONLY output data structures. Defines the exact shape
of per-frame and session-level output that this module hands off to
other developers/teams. No computation happens here — this file just
describes the contract.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


@dataclass
class FrameFeatures:
    """
    One frame's worth of extracted eye features. This is what
    EyeCaptureModel.process_frame() will return in the next stage.
    """
    timestamp: float
    face_detected: bool

    # EAR values — None if no face detected this frame.
    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    mean_ear: Optional[float] = None

    # True only on the exact frame a blink was confirmed to have completed.
    blink_detected: bool = False

    # Eye movement — horizontal is the reliable signal (Stage 7 validated).
    eye_movement_horizontal_label: Optional[str] = None   # "outward" | "inward" | "center"
    eye_movement_horizontal_ratio: Optional[float] = None

    # Vertical is explicitly low-confidence — see Known Issues in tracker.
    eye_movement_vertical_label: Optional[str] = None     # "up" | "down" | "center"
    eye_movement_vertical_ratio: Optional[float] = None
    eye_movement_vertical_confidence: str = "low"

    # Quality — whether THIS frame's data should be trusted at all.
    quality_valid: bool = False
    quality_score: float = 0.0
    quality_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain-dict/JSON-serializable form."""
        return asdict(self)


@dataclass
class SessionFeatures:
    """
    Full-session summary statistics. This is what
    EyeCaptureModel.get_session_features() will return.
    """
    session_duration_sec: float
    total_frames: int
    valid_frames: int
    valid_frames_pct: float  # quality-valid frames / total frames * 100

    mean_ear: float
    ear_variance: float

    blink_count: int
    blink_rate_per_min: float
    mean_blink_duration_ms: float

    eye_movement_horizontal_distribution_pct: Dict[str, float]
    eye_movement_vertical_distribution_pct: Dict[str, float]
    eye_movement_vertical_confidence: str = "low"

    def to_dict(self) -> dict:
        return asdict(self)