"""
Stage 6 test: runs live blink detection, displaying current EAR,
blink state, total blink count, and last blink duration.

Run: python blink_test.py   (from inside src/)
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


def _standalone_test():
    stream = CameraStream(camera_index=0)
    if not stream.open():
        print("Exiting: camera failed to open.")
        return

    detector = FaceDetector(model_path=MODEL_PATH)
    blink_detector = BlinkDetector()
    print("Press 'q' to quit.")

    while True:
        success, frame, timestamp = stream.read_frame()
        if not success:
            continue

        landmarks = detector.detect(frame)

        if landmarks is not None:
            ears = calculate_both_ears(landmarks, RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX)
            mean_ear = ears["mean_ear"]

            blink_event = blink_detector.update(mean_ear, timestamp)

            cv2.putText(frame, f"Mean EAR: {mean_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_detector.total_blinks}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if blink_detector.last_blink is not None:
                cv2.putText(frame,
                            f"Last blink duration: {blink_detector.last_blink.duration_sec*1000:.0f} ms",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if blink_event is not None:
                print(f"Blink detected! Duration: {blink_event.duration_sec*1000:.0f} ms")
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stage 6 - Blink Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _standalone_test()