"""
landmark_visualizer.py

Stage 3 test: draws only the eye-region landmark points defined in
eye_regions.py, to visually confirm the indices line up with actual
eye corners/lids before we build math on top of them.

Run: python landmark_visualizer.py   (from inside src/)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")

import cv2
from camera import CameraStream
from face_detector import FaceDetector
from eye_regions import EYE_LANDMARKS

EYE_COLORS = {
    "right": (0, 255, 255),  # yellow
    "left": (255, 0, 255),   # magenta
}


def draw_eye_landmarks(frame, landmarks, frame_width, frame_height):
    for eye_name, indices in EYE_LANDMARKS.items():
        color = EYE_COLORS[eye_name]
        for idx in indices:
            lm = landmarks[idx]
            x_px = int(lm.x * frame_width)
            y_px = int(lm.y * frame_height)
            cv2.circle(frame, (x_px, y_px), 2, color, -1)


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
            draw_eye_landmarks(frame, landmarks, w, h)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stage 3 - Eye Landmarks Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _standalone_test()