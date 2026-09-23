"""
Stage 10 test: runs the complete pipeline (detection -> EAR -> blink ->
movement -> quality) and packs every frame into a FrameFeatures record,
then builds a SessionFeatures summary at the end and saves both to JSON
in outputs/.

Run: python schema_test.py   (from inside src/)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")

import cv2
import json
import os
from camera import CameraStream
from face_detector import FaceDetector
from eye_regions import RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX
from ear import calculate_both_ears
from blink_detector import BlinkDetector
from gaze_estimator import estimate_eye_movement
from feature_aggregator import SessionAggregator
from quality import validate_frame
from schema import FrameFeatures, SessionFeatures


def _standalone_test():
    stream = CameraStream(camera_index=0)
    if not stream.open():
        print("Exiting: camera failed to open.")
        return

    detector = FaceDetector(model_path=MODEL_PATH)
    blink_detector = BlinkDetector()
    aggregator = SessionAggregator()

    all_frame_records = []  # optional: keep every frame's record for export
    valid_frame_count = 0
    total_frame_count = 0

    print("Recording session. Press 'q' to end and save output.")

    while True:
        success, frame, timestamp = stream.read_frame()
        if not success:
            continue

        landmarks = detector.detect(frame)
        h, w = frame.shape[:2]
        total_frame_count += 1

        if landmarks is not None:
            ears = calculate_both_ears(landmarks, RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX)
            mean_ear = ears["mean_ear"]
            movement = estimate_eye_movement(landmarks)

            bbox = detector.get_bounding_box(landmarks, w, h)
            quality = validate_frame(landmarks, frame, bbox)

            blink_event = blink_detector.update(mean_ear, timestamp)

            # Only feed valid frames into session-level aggregation —
            # this is the actual point of Stage 9's quality gating.
            if quality.is_valid:
                aggregator.update(mean_ear, timestamp, movement)
                valid_frame_count += 1

            record = FrameFeatures(
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

            status_text = "VALID" if quality.is_valid else "INVALID: " + ",".join(quality.reasons)
            cv2.putText(frame, f"EAR:{mean_ear:.2f} Blinks:{blink_detector.total_blinks} {status_text}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            record = FrameFeatures(timestamp=timestamp, face_detected=False,
                                    quality_reasons=["no_face_detected"])
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        all_frame_records.append(record)

        cv2.imshow("Stage 10 - Output Schema Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()

    # --- Build session summary ---
    agg_stats = aggregator.get_session_features(blink_detector)
    valid_pct = round(100.0 * valid_frame_count / total_frame_count, 1) if total_frame_count > 0 else 0.0

    session = SessionFeatures(
        session_duration_sec=agg_stats.get("session_duration_sec", 0.0),
        total_frames=total_frame_count,
        valid_frames=valid_frame_count,
        valid_frames_pct=valid_pct,
        mean_ear=agg_stats.get("mean_ear", 0.0),
        ear_variance=agg_stats.get("ear_variance", 0.0),
        blink_count=agg_stats.get("blink_count", 0),
        blink_rate_per_min=agg_stats.get("blink_rate_per_min", 0.0),
        mean_blink_duration_ms=agg_stats.get("mean_blink_duration_ms", 0.0),
        eye_movement_horizontal_distribution_pct=agg_stats.get("eye_movement_horizontal_distribution_pct", {}),
        eye_movement_vertical_distribution_pct=agg_stats.get("eye_movement_vertical_distribution_pct", {}),
    )

    # --- Save to outputs/ ---
    os.makedirs("../outputs", exist_ok=True)
    output_path = "../outputs/session_output.json"
    with open(output_path, "w") as f:
        json.dump({
            "session_summary": session.to_dict(),
            "frame_count_exported": len(all_frame_records),
        }, f, indent=2)

    print(f"\nSession summary saved to {output_path}")
    print(json.dumps(session.to_dict(), indent=2))


if __name__ == "__main__":
    _standalone_test()