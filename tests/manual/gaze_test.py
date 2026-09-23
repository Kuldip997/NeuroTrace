"""
Stage 7 test: displays live iris-position estimates for both eyes.

Run from anywhere, e.g.: python tests/manual/gaze_test.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")

import cv2
from camera import CameraStream
from face_detector import FaceDetector
from gaze_estimator import estimate_eye_movement


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
            movement = estimate_eye_movement(landmarks)
            r = movement["right_eye"]
            l = movement["left_eye"]

            cv2.putText(frame, f"Right: {r['horizontal_label']} / {r['vertical_label']} "
                                f"({r['horizontal_ratio']:.2f}, {r['vertical_ratio']:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Left:  {l['horizontal_label']} / {l['vertical_label']} "
                                f"({l['horizontal_ratio']:.2f}, {l['vertical_ratio']:.2f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stage 7 - Eye Movement Estimation Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _standalone_test()