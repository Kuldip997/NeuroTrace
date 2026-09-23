"""
Stage 8 test: runs the full pipeline live, feeding every frame into
SessionAggregator, and prints the session summary when you quit ('q').

Run: python aggregator_test.py   (from inside src/)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")

import cv2
from camera import CameraStream
from face_detector import FaceDetector
from eye_regions import RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX
from ear import calculate_both_ears
from blink_detector import BlinkDetector
from gaze_estimator import estimate_eye_movement
from feature_aggregator import SessionAggregator


def _standalone_test():
    stream = CameraStream(camera_index=0)
    if not stream.open():
        print("Exiting: camera failed to open.")
        return

    detector = FaceDetector(model_path=MODEL_PATH)
    blink_detector = BlinkDetector()
    aggregator = SessionAggregator()

    print("Recording session. Blink, look around, etc. Press 'q' to end and see summary.")

    while True:
        success, frame, timestamp = stream.read_frame()
        if not success:
            continue

        landmarks = detector.detect(frame)

        if landmarks is not None:
            ears = calculate_both_ears(landmarks, RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX)
            mean_ear = ears["mean_ear"]
            movement = estimate_eye_movement(landmarks)

            blink_detector.update(mean_ear, timestamp)
            aggregator.update(mean_ear, timestamp, movement)

            cv2.putText(frame, f"Mean EAR: {mean_ear:.3f}  Blinks: {blink_detector.total_blinks}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stage 8 - Session Aggregation Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()

    # Print the session summary after the window closes.
    summary = aggregator.get_session_features(blink_detector)
    print("\n--- SESSION SUMMARY ---")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    _standalone_test()