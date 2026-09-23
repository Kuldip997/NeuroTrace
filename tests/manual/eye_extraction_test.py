"""
Stage 4 test: extracts and displays both eye crops as separate
thumbnail windows, alongside the main feed with bounding boxes drawn.

Run: python eye_extraction_test.py   (from inside src/)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")

import cv2
from camera import CameraStream
from face_detector import FaceDetector
from eye_regions import extract_both_eyes

BOX_COLORS = {
    "right": (0, 255, 255),  # yellow
    "left": (255, 0, 255),   # magenta
}


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
            eyes = extract_both_eyes(landmarks, frame)

            for eye_name, data in eyes.items():
                x_min, y_min, x_max, y_max = data["bbox"]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              BOX_COLORS[eye_name], 1)

                crop = data["crop"]
                if crop.size > 0:
                    # Resize for a consistent, readable thumbnail size
                    thumb = cv2.resize(crop, (160, 100))
                    cv2.imshow(f"{eye_name} eye crop", thumb)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stage 4 - Eye Extraction Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _standalone_test()