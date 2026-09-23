"""
Stage 5 test: displays live EAR values (left, right, mean) on screen
as text, so you can watch the numbers change in real time — open eyes
vs. blinking vs. squinting.

Run: python ear_test.py   (from inside src/)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")

import cv2
from camera import CameraStream
from face_detector import FaceDetector
from eye_regions import RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX
from ear import calculate_both_ears


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

        if landmarks is not None:
            ears = calculate_both_ears(landmarks, RIGHT_EYE_EAR_IDX, LEFT_EYE_EAR_IDX)

            cv2.putText(frame, f"Right EAR: {ears['right_ear']:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Left EAR:  {ears['left_ear']:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Mean EAR:  {ears['mean_ear']:.3f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stage 5 - EAR Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _standalone_test()