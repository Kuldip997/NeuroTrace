"""
Stage 9 test: shows live quality validation status — valid/invalid,
score, and specific failure reasons.

Run: python quality_test.py   (from inside src/)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")

import cv2
from camera import CameraStream
from face_detector import FaceDetector
from quality import validate_frame


def _standalone_test():
    stream = CameraStream(camera_index=0)
    if not stream.open():
        print("Exiting: camera failed to open.")
        return

    detector = FaceDetector(model_path=MODEL_PATH)
    print("Press 'q' to quit.")

    while True:
        success, frame, _ = stream.read_frame()
        if not success:
            continue

        landmarks = detector.detect(frame)
        h, w = frame.shape[:2]

        if landmarks is not None:
            bbox = detector.get_bounding_box(landmarks, w, h)
            result = validate_frame(landmarks, frame, bbox)

            color = (0, 255, 0) if result.is_valid else (0, 0, 255)
            status = "VALID" if result.is_valid else "INVALID"
            cv2.putText(frame, f"{status}  score={result.quality_score:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if result.reasons:
                cv2.putText(frame, ", ".join(result.reasons), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "INVALID  no_face_detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Stage 9 - Quality Validation Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _standalone_test()