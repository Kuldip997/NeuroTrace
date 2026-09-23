# NeuroTrace Eye Capture Model — Project Tracker

## 1. Project Overview

### Project Name
NeuroTrace — Standalone Eye Capture Model

### Purpose
A standalone, webcam-based module that captures eye-related behavioral
features (blinks, eye openness, coarse eye-movement direction, frame
quality) as structured numerical output, for later integration into
the larger NeuroTrace multimodal system by other team members.

### Scope
- Webcam frame capture
- Face + landmark detection (MediaPipe FaceLandmarker)
- Eye Aspect Ratio (EAR) calculation
- Temporal blink detection (state machine)
- Coarse eye-movement estimation (iris-relative-to-eye-corner position)
- Frame-level quality validation (head pose, lighting, blur, face size)
- Session-level feature aggregation
- Structured per-frame and session output schema
- Public `EyeCaptureModel` class as the integration interface

### Out of Scope
- Speech/audio analysis
- Handwriting analysis
- Multimodal fusion
- Final NeuroTrace decision system
- Medical/clinical diagnosis of any kind
- Full application/dashboard
- Database/backend integration
- True screen-coordinate gaze tracking (would require head-pose
  correction + per-user calibration — explicitly excluded)

**Status: Functionally complete, per this scope.**

---

## 2. Current Architecture

```text
Camera (caller-provided frame, e.g. via CameraStream)
  ↓
Frame Acquisition        (OpenCV VideoCapture, frame timestamping)
  ↓
Face Detection + Landmarks (MediaPipe FaceLandmarker — single pass)
  ↓
Eye Region Extraction     (slice landmark indices for L/R eye + iris)
  ↓
Per-Frame Feature Calc    (EAR L/R/mean, iris-relative position)
  ↓
Temporal State Machine    (blink open/closing/closed/opening states, duration)
  ↓
Quality Validation        (face/eye visibility, head pose proxy, lighting, blur, face size)
  ↓
Feature Aggregation       (valid frames only → rolling + full-session stats)
  ↓
Structured Output         (FrameFeatures per frame, SessionFeatures per session)
  ↓
Public Interface          (EyeCaptureModel: process_frame / get_session_features / reset / close)
```

---

## 3. Technology Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.11.5 | Runtime |
| Vision | opencv-python | 5.0.0 | Camera capture, frame ops, drawing, blur/brightness checks |
| Landmarks | mediapipe | 1.0.1 | Face + iris landmark detection (Tasks API — `mp.solutions` does NOT exist in this version, confirmed by direct inspection) |
| Numerical | numpy | 2.4.6 | Distance/array math |
| Data | pandas | 3.0.5 | Installed, reserved for future CSV export — not currently used |

Note: `mediapipe 1.0.1` depends on `opencv-contrib-python` internally,
which installs alongside `opencv-python`. Both present at matching
versions is expected and correct — do not remove either.

---

## 4. Feature List

| Feature | Status | Method | File |
|---|---|---|---|
| Webcam capture | ✅ Complete | OpenCV VideoCapture | `src/camera.py` |
| Face + landmark detection | ✅ Complete | MediaPipe FaceLandmarker (VIDEO mode) | `src/face_detector.py` |
| Eye region extraction (crop) | ✅ Complete | Landmark bounding box + padding | `src/eye_regions.py` |
| EAR (left/right/mean) | ✅ Complete | Standard 6-point EAR formula | `src/ear.py` |
| Blink detection (event) | ✅ Complete | Temporal state machine | `src/blink_detector.py` |
| Blink count / duration / history | ✅ Complete | `BlinkDetector.blink_events` | `src/blink_detector.py` |
| Eye movement — horizontal | ✅ Complete, validated reliable | Iris-position ratio vs eye corners | `src/gaze_estimator.py` |
| Eye movement — vertical | ⚠️ Experimental (low confidence, by design) | Iris-position ratio vs lid points | `src/gaze_estimator.py` |
| Quality validation | ✅ Complete | Head pose proxy, brightness, sharpness, face size | `src/quality.py` |
| Session feature aggregation | ✅ Complete | Runs only on quality-valid frames | `src/feature_aggregator.py` |
| Output schema | ✅ Complete | `FrameFeatures`, `SessionFeatures` dataclasses | `src/schema.py` |
| Public `EyeCaptureModel` API | ✅ Complete | Wraps entire pipeline | `src/eye_capture_model.py` |
| Reference usage example | ✅ Complete | Demonstrates integration using only 2 imports | `main.py` |

---

## 5. Project File Structure

```text
eye-capture/
│
├── venv/
├── models/
│   └── face_landmarker.task
├── src/
│   ├── camera.py
│   ├── face_detector.py
│   ├── eye_regions.py
│   ├── ear.py
│   ├── blink_detector.py
│   ├── gaze_estimator.py
│   ├── feature_aggregator.py
│   ├── quality.py
│   ├── schema.py
│   └── eye_capture_model.py
├── tests/
│   └── manual/
│       ├── landmark_visualizer_test.py
│       ├── eye_extraction_test.py
│       ├── ear_test.py
│       ├── blink_test.py
│       ├── gaze_test.py
│       ├── aggregator_test.py
│       ├── quality_test.py
│       └── schema_test.py
├── data/                    (reserved, empty — for future recorded test videos)
├── outputs/
│   └── session_output.json
├── requirements.txt
├── PROJECT_TRACKER.md
└── main.py                  (integration reference — imports only camera.py + eye_capture_model.py)
```

Not created (optional, not required for current scope): `configs/`
(thresholds currently live as named constants in their own files,
satisfying the "no magic numbers" rule already).

---

## 6. File-by-File Responsibilities

| File | Responsibility | Status |
|---|---|---|
| `src/camera.py` | Webcam frame acquisition, FPS tracking | Complete |
| `src/face_detector.py` | MediaPipe FaceLandmarker wrapper (VIDEO mode) | Complete |
| `src/eye_regions.py` | Eye/iris landmark index constants, bounding box + crop extraction | Complete |
| `src/ear.py` | Pure EAR math functions | Complete |
| `src/blink_detector.py` | Temporal blink state machine, full blink history | Complete |
| `src/gaze_estimator.py` | Coarse iris-relative eye movement (horizontal reliable, vertical experimental) | Complete but flagged |
| `src/quality.py` | Per-frame validity checks (pose proxy, lighting, blur, face size) | Complete |
| `src/feature_aggregator.py` | Session-level statistics from quality-valid frames | Complete |
| `src/schema.py` | `FrameFeatures` / `SessionFeatures` output data structures | Complete |
| `src/eye_capture_model.py` | Public interface: `process_frame`, `get_session_features`, `reset`, `close` | Complete |
| `main.py` | Reference integration example; entry point for running the whole model | Complete |

**Files that should NOT be redesigned without a strong reason:** the
core pipeline order (detect → EAR → blink → movement → quality →
aggregate) and the `EyeCaptureModel` public method signatures — other
NeuroTrace components may already be planned against this interface.

---

## 7. Completed Work

### Step 1 — Environment Setup
- Python 3.11.5 venv created (rebuilt once — original had leftover
  unrelated packages from a non-fresh venv, plus a corrupted `cv2`
  install from `opencv-python` + `opencv-contrib-python` conflict)
- Confirmed mediapipe 1.0.1 no longer pins `numpy<2`

### Stage 1 — Webcam Capture
- `camera.py` — live feed + FPS overlay, clean release on 'q'. Actual camera resolution 1280x720 (webcam overrides the 640x480 request; harmless since all math uses normalized coordinates).

### Stage 2 — Face Detection
- `face_detector.py` using MediaPipe Tasks API (`vision.FaceLandmarker`)
- Confirmed `mp.solutions` does not exist in mediapipe 1.0.1 — Tasks API is mandatory on this version
- Model file `face_landmarker.task` (no blendshapes variant) downloaded and verified

### Stage 3 — Eye Landmarks
- Eye landmark indices defined in `eye_regions.py`; visually confirmed correct placement on eye corners/lids

### Stage 4 — Eye Region Extraction
- Bounding box + crop logic added to `eye_regions.py`; confirmed via 3-window test

### Stage 5 — EAR Implementation
- `ear.py` built with standard 6-point EAR formula; confirmed stable open-eye values, clear drop on blink
- Kuldeep's personal baseline open-eye mean EAR runs notably higher (~0.43–0.57) than the generic reference figure (~0.25–0.35) — this is normal person-to-person/camera-angle variation, not a bug

### Stage 6 — Blink Detection
- `blink_detector.py` — state machine (OPEN/CLOSED transitions, min consecutive frames, min/max duration validation)
- Confirmed accurate single-count-per-blink, correctly rejects long closures (>0.5s)

### Stage 7 — Eye Movement Estimation
- `gaze_estimator.py` using iris landmarks (indices 468/473 centers)
- Horizontal estimation validated reliable and consistent across both eyes
- Vertical estimation confirmed unreliable/can invert under head pitch + eyelid narrowing — downgraded to experimental (see Known Issues)

### Stage 8 — Temporal Feature Aggregation
- `feature_aggregator.py` — session-level stats (blink count/rate/duration, EAR mean/variance, movement label distributions)
- Bug found and fixed: `BlinkDetector` initially only stored `last_blink`; added `blink_events` list to support session-wide stats
- Initial uncontrolled test showed an implausible 64 blinks/min — investigated via two controlled tests (blink-only, head-movement-only). Result: blink logic itself was accurate (13 blinks matched deliberate count); head movement alone did NOT reliably cause false blinks, disproving the initial "head yaw distorts EAR into false blinks" hypothesis. Real cause of the 64/min reading was genuine rapid blinking during that specific unguided test, not a bug.

### Stage 9 — Quality Validation
- `quality.py` — checks: no-face, head yaw/pitch (2D geometric proxy, NOT true 3D pose), poor lighting (brightness), motion blur (Laplacian variance), face-too-small/far
- All 7 manual test scenarios (straight-on, yaw, pitch, poor lighting, blur, distance, out-of-frame) confirmed correctly flipping VALID/INVALID
- Confirmed real-world effectiveness: gating blink/aggregation to valid-only frames brought blink rate from an implausible 64/min down to a physiologically normal ~14–22/min in later tests

### Stage 10 — Output Schema
- `schema.py` — `FrameFeatures` and `SessionFeatures` dataclasses, both with `to_dict()` for JSON export
- `eye_movement_vertical_confidence: "low"` field carried through to final output, confirmed present in saved JSON
- Validated with two real sessions: mixed movement/blur session → 50.3% valid frames (expected, given deliberate bad-condition testing); still-and-blinking baseline session → 85% valid frames, 21.9 blinks/min, both within normal ranges

### Final Stage — Public Interface
- `eye_capture_model.py` — `EyeCaptureModel` class wrapping the full pipeline behind `process_frame()`, `get_session_features()`, `reset()`, `close()`
- Reference usage script (`model_usage_example.py` → later moved to `main.py`) built importing ONLY `camera.py` and `eye_capture_model.py`, proving the abstraction boundary holds

### Project Cleanup / Reorganization
- Moved all 8 stage-verification scripts from `src/` into `tests/manual/`, renamed for consistency (`landmark_visualizer.py` → `landmark_visualizer_test.py`)
- Moved `model_usage_example.py` to project-root `main.py`
- **Path bug found and fixed twice** during this move: relative model paths (`"../models/..."`, `"models/..."`) break depending on which directory the script is run from. Fixed with a directory-independent pattern using `__file__`:
```python
  import sys, os
  sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
  MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")
```
  (path depth adjusted per file's actual location — 2 levels up from `tests/manual/`, 1 level up... actually 0 extra levels from project-root `main.py`, since `models/` sits directly alongside it there).
  **This pattern should be reused for any future file added to this project that loads the model** — never hardcode a relative path string.

---

## 8. Current Implementation State

```text
Current Stage:
Project complete through defined scope.

Completed:
- Full pipeline: capture → detect → EAR → blink → movement → quality → aggregate → structured output
- Public EyeCaptureModel interface, validated via main.py
- Project reorganized into clean src/ + tests/manual/ + root main.py structure
- Directory-independent model path resolution applied project-wide

Currently Working On:
- Nothing active — awaiting further direction (more test scenarios,
  config centralization, or handoff to NeuroTrace integration).

Next (optional, not required for current scope):
- Additional test scenarios: glasses, multiple faces, low light edge cases, face partially obstructed
- Optional: configs/default_config.yaml to centralize tunable thresholds
- Optional: CSV export via pandas (installed, unused so far)
- Handoff to NeuroTrace multimodal fusion team
```

---

## 9. Important Technical Decisions

### Decision: MediaPipe FaceLandmarker (Tasks API), not legacy `mp.solutions.face_mesh`
Reason: `mp.solutions` is fully removed in mediapipe 1.0.1 (confirmed by direct inspection). No legacy fallback exists on this version.

### Decision: `face_landmarker.task` (no blendshapes) model
Reason: Facial expression coefficients aren't needed. The 478 landmarks including iris are output by the base model regardless of the blendshapes flag.

### Decision: VIDEO running mode with manually incrementing timestamp
Reason: Simpler than LIVE_STREAM's async callback pattern for a single-threaded synchronous webcam loop.

### Decision: EAR-based blink detection via temporal state machine, not per-frame thresholding
Reason: Per-frame thresholding massively over-counts blinks. State machine with min-consecutive-frames and min/max duration filters gives accurate single-count blinks.

### Decision: Vertical eye-movement kept as experimental/low-confidence output, not removed
Reason: Confirmed via live testing that vertical ratio can invert direction when head pitch and eyelid narrowing combine during downward gaze. Correct fix needs 3D head-pose correction — explicitly out of scope. Kept in output but clearly flagged rather than silently unreliable.

### Decision: Quality validation gates what enters session aggregation, not just a display flag
Reason: Controlled testing showed ungated aggregation produced a physiologically implausible 64 blinks/min from noisy frames; gating to `quality.is_valid` frames only brought this to normal (~14-22/min) ranges. This is the single most impactful correctness decision in the project.

### Decision: Head yaw/pitch checks use a 2D landmark-ratio proxy, not MediaPipe's 3D transformation matrices
Reason: `output_facial_transformation_matrixes` was deliberately left off in Stage 2 since true 3D pose wasn't needed elsewhere; the 2D proxy is a deliberate lightweight trade-off, good enough to flag "clearly turned away" without full 3D pose computation.

### Decision: `EyeCaptureModel` does not manage its own camera
Reason: Keeps it usable regardless of frame source (live webcam, video file, or another NeuroTrace component's capture loop) — it only processes frames handed to it via `process_frame()`.

### Decision: Test scripts use directory-independent path resolution (`__file__`-based), not hardcoded relative paths
Reason: Hardcoded relative paths (`"../models/..."` or `"models/..."`) broke depending on the directory the script was launched from, causing two separate `FileNotFoundError` incidents during the Stage 11 reorganization. Fixed project-wide.

---
### Decision: EyeCaptureModel's default model_path also uses __file__-based resolution
Reason: The default was initially a plain relative string
("models/face_landmarker.task"), which is the same fragile pattern
that caused real FileNotFoundError bugs in the test scripts. Even
though every current caller passes model_path explicitly, an unused
but broken default is still a latent bug. Fixed for consistency with
the rest of the project.

## 10. Configuration / Parameters

```text
EAR_THRESHOLD: 0.21
MIN_CONSEC_CLOSED_FRAMES: 2
MIN_BLINK_DURATION_SEC: 0.05
MAX_BLINK_DURATION_SEC: 0.5
EYE_CROP_PADDING_RATIO: 0.4
HORIZONTAL_LEFT_THRESHOLD: 0.35
HORIZONTAL_RIGHT_THRESHOLD: 0.65
VERTICAL_UP_THRESHOLD: 0.35   (unreliable — see Known Issues)
VERTICAL_DOWN_THRESHOLD: 0.65 (unreliable — see Known Issues)
YAW_RATIO_MIN / MAX: 0.35 / 0.65
PITCH_RATIO_MIN / MAX: 0.35 / 0.65
MIN_BRIGHTNESS: 40.0
MIN_SHARPNESS: 50.0
MIN_FACE_WIDTH_RATIO: 0.15
Camera resolution: 640x480 requested, 1280x720 actual (webcam-dependent)
Face detection/tracking confidence: 0.5 / 0.5 (MediaPipe defaults)
```

No parameter changes made from initial defaults — all validated as reasonable via live testing (Stages 5-10), not yet needing per-user tuning.

---

## 11. Known Issues

```text
Issue 1:
Vertical eye-movement estimation can report the OPPOSITE of actual
direction when head pitch changes alongside gaze (downward gaze +
head tilt narrows the eyelid opening, distorting the ratio nonlinearly).

Status: Known limitation, confirmed via live testing (Stage 7).
Fix would require head-pose correction — out of scope.
Mitigation: output tagged with eye_movement_vertical_confidence="low"
throughout the schema. Horizontal remains the trusted movement signal.

---

Issue 2:
Head yaw/pitch quality checks use a 2D landmark-ratio proxy, not true
3D head pose. Good enough to catch "clearly turned away" but not a
precise angle measurement.

Status: Accepted design trade-off, not a bug. Documented in quality.py.

---

Issue 3 (resolved, kept for history):
Initial hypothesis that head-turning directly causes false-positive
blinks (by shrinking the EAR horizontal denominator) was tested via
controlled experiment and DISPROVEN — head movement alone did not
reliably cross the blink threshold. The real cause of one anomalously
high blink-rate reading was genuine rapid blinking during an unguided
test, not a systematic bug. Do not re-introduce yaw-based EAR
correction based on this old hypothesis without new evidence.
```

---

## 12. Testing Status

| Test | Status | Result | Notes |
|---|---|---|---|
| Webcam capture | ✅ | Passed | Stable FPS, clean release |
| Face detection | ✅ | Passed | Accurate bounding box tracking |
| Eye landmarks | ✅ | Passed | Dots land correctly on eyes |
| Eye extraction/crop | ✅ | Passed | Clean thumbnails, correct L/R labeling |
| EAR values | ✅ | Passed | Stable open, drops sharply on blink |
| Blink detection | ✅ | Passed | Accurate single-count, rejects long closures |
| Blink rate (controlled, still) | ✅ | Passed | 13 blinks / 25s matched deliberate count |
| Blink rate (controlled, head movement only) | ✅ | Passed | Only 1 false blink in 39s of movement |
| Eye movement — horizontal | ✅ | Passed | Consistent, symmetric, both eyes agree |
| Eye movement — vertical | ⚠️ | Unreliable (by design, documented) | Inverts under head pitch + narrowed lid |
| Quality: straight-on | ✅ | Passed | VALID, high score |
| Quality: head yaw | ✅ | Passed | Correctly flags `head_yaw_too_extreme` |
| Quality: head pitch | ✅ | Passed | Correctly flags `head_pitch_too_extreme` |
| Quality: poor lighting | ✅ | Passed | Correctly flags `poor_lighting` |
| Quality: motion blur | ✅ | Passed | Correctly flags `motion_blur` |
| Quality: face too far | ✅ | Passed | Correctly flags `face_too_small_or_far` |
| Quality: no face | ✅ | Passed | Correctly flags `no_face_detected` |
| Session aggregation (mixed conditions) | ✅ | Passed | 50.3% valid — expected given deliberate bad-condition testing |
| Session aggregation (baseline, still) | ✅ | Passed | 85% valid, 21.9 blinks/min — physiologically normal |
| Full pipeline via `EyeCaptureModel` | ✅ | Passed | `main.py` runs cleanly end-to-end |
| Glasses | ⏳ | Not yet tested | — |
| Multiple faces | ⏳ | Not yet tested | — |
| Low light (severe, not just dim) | ⏳ | Not yet tested | — |

---

## 13. Output Schema

```python
# Per-frame (FrameFeatures)
{
  "timestamp": float,
  "face_detected": bool,
  "left_ear": float | null,
  "right_ear": float | null,
  "mean_ear": float | null,
  "blink_detected": bool,
  "eye_movement_horizontal_label": "outward" | "inward" | "center" | null,
  "eye_movement_horizontal_ratio": float | null,
  "eye_movement_vertical_label": "up" | "down" | "center" | null,
  "eye_movement_vertical_ratio": float | null,
  "eye_movement_vertical_confidence": "low",
  "quality_valid": bool,
  "quality_score": float,
  "quality_reasons": [string]
}

# Session-level (SessionFeatures)
{
  "session_duration_sec": float,
  "total_frames": int,
  "valid_frames": int,
  "valid_frames_pct": float,
  "mean_ear": float,
  "ear_variance": float,
  "blink_count": int,
  "blink_rate_per_min": float,
  "mean_blink_duration_ms": float,
  "eye_movement_horizontal_distribution_pct": {"outward": float, "inward": float, "center": float},
  "eye_movement_vertical_distribution_pct": {"up": float, "down": float, "center": float},
  "eye_movement_vertical_confidence": "low"
}
```

Both defined as dataclasses in `src/schema.py` with `.to_dict()` for JSON serialization. Confirmed matches actual saved output in `outputs/session_output.json`.

---

## 14. Dependencies

Matches `requirements.txt`:
```text
opencv-python==5.0.0.93
opencv-contrib-python==5.0.0.93
mediapipe==1.0.1
numpy==2.4.6
pandas==3.0.5
(plus transitive deps: absl-py, cffi, contourpy, cycler, flatbuffers,
fonttools, kiwisolver, matplotlib, packaging, pillow, pycparser,
pyparsing, python-dateutil, six, sounddevice, tzdata)
```

---

## 15. How to Run

```bash
cd eye-capture
venv\Scripts\Activate.ps1

# Run the complete model (recommended entry point):
python main.py

# Run an individual stage-verification script:
python tests\manual\blink_test.py
python tests\manual\quality_test.py
# etc. — all 8 scripts in tests/manual/ work from any directory
# since they resolve the model path via __file__, not a hardcoded string.
```

---

## 16. What the Next AI Should Know

> This project is functionally complete through its defined scope: a
> standalone webcam-based Eye Capture Model producing EAR, blink,
> coarse eye-movement, and quality-gated session statistics, exposed
> via the `EyeCaptureModel` class in `src/eye_capture_model.py`.
>
> Do not redesign the pipeline order or the `EyeCaptureModel` public
> method signatures (`process_frame`, `get_session_features`, `reset`,
> `close`) without a strong technical reason — other NeuroTrace
> components may be planned against this interface.
>
> Known, accepted limitations (do not "fix" without new evidence):
> vertical eye-movement is low-confidence by design (2D iris-ratio
> limitation under head pitch); head yaw/pitch quality checks are a 2D
> geometric proxy, not true 3D pose. Quality gating (only feeding
> `quality.is_valid` frames into session aggregation) is essential to
> correctness — do not aggregate ungated frames, this was confirmed to
> produce implausible blink rates during testing.
>
> Environment: Python 3.11.5, mediapipe 1.0.1 (Tasks API only —
> `mp.solutions` does not exist on this version, do not suggest legacy
> MediaPipe solutions API).
>
> Any new script that loads the model file must use the
> `__file__`-based path resolution pattern (see Section 9, last
> decision) — hardcoded relative paths caused two real bugs during
> this project's reorganization.
>
> Suggested next steps if development continues: test glasses/multiple-
> faces/severe-low-light scenarios (Section 12), optionally centralize
> thresholds into a config file, optionally add CSV export via the
> already-installed pandas dependency, then hand off to the NeuroTrace
> multimodal fusion team.

---

## 17. Change Log

```text
## [2026-09-17] — Initial Tracker Creation

### Changes
- Created PROJECT_TRACKER.md after completing Stages 1-7

### Files Changed
- PROJECT_TRACKER.md (created)

### Testing
- All Stages 1-7 manually verified via live webcam testing

### Result
- Core detection/EAR/blink/movement pipeline functional
- One known limitation documented (vertical gaze)

### Next Step
- Stage 8: temporal feature aggregation

---

## [2026-09-21] — Project Completion & Reorganization

### Changes
- Completed Stage 8 (session aggregation), Stage 9 (quality validation),
  Stage 10 (output schema), and the final public EyeCaptureModel interface
- Fixed a real bug: BlinkDetector was missing blink_events history
  (crashed feature_aggregator.py on first use)
- Investigated and resolved an implausible 64-blinks/min reading via
  controlled testing; confirmed root cause was genuine rapid blinking,
  not a systematic bug — original head-yaw hypothesis was disproven
- Confirmed quality gating brings blink rate to physiologically normal
  ranges (14-22/min) in properly-conducted sessions
- Reorganized project: moved 8 stage scripts into tests/manual/,
  renamed for consistency, moved model_usage_example.py to root main.py
- Fixed relative-model-path bugs (twice) during reorganization using
  __file__-based path resolution — applied project-wide

### Files Changed
- src/blink_detector.py (added blink_events)
- src/feature_aggregator.py, src/quality.py, src/schema.py,
  src/eye_capture_model.py (created)
- main.py (created, at root)
- tests/manual/*.py (8 files moved, renamed, path-fixed)
- PROJECT_TRACKER.md (this update)

### Testing
- All stages 1-10 manually verified via live webcam testing, including
  two controlled experiments for blink-rate diagnosis and two full
  session runs validating output schema correctness

### Result
- Standalone Eye Capture Model complete and functional per original scope

### Next Step
- None required. Optional: expand test coverage (glasses, multiple
  faces, severe low light), config centralization, or hand off to
  NeuroTrace integration.

---

## [2026-09-21] — Final Verification Pass

### Changes
- Confirmed main.py runs correctly end-to-end after reorganization
  (previously described but not actually re-tested)
- Fixed EyeCaptureModel's default model_path parameter to use
  __file__-based resolution instead of a plain relative string,
  closing a latent (unexercised but real) path bug

### Files Changed
- src/eye_capture_model.py
- PROJECT_TRACKER.md (this update)

### Testing
- python main.py confirmed working from project root

### Result
- Both previously-flagged verification gaps closed
- Project confirmed complete through defined scope, with no open items

### Next Step
- None required for current scope. Future work (optional): glasses/
  multiple-face/severe-low-light testing, config centralization, CSV
  export via pandas, handoff to NeuroTrace multimodal fusion team.
```
