"""
model_usage_example.py

Demonstrates how another developer would use EyeCaptureModel WITHOUT
needing to know anything about MediaPipe, EAR, or blink detection
internals. This is the reference example for teammates integrating
this module into the larger NeuroTrace system.

Run: python model_usage_example.py   (from inside src/)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import cv2
import json
from camera import CameraStream
from eye_capture_model import EyeCaptureModel


def main():
    stream = CameraStream(camera_index=0)
    if not stream.open():
        print("Camera failed to open.")
        return

    model = EyeCaptureModel(model_path="models/face_landmarker.task")

    print("Press 'q' to end session and print summary.")

    while True:
        success, frame, timestamp = stream.read_frame()
        if not success:
            continue

        features = model.process_frame(frame, timestamp)

        if features.face_detected:
            status = "VALID" if features.quality_valid else "INVALID"
            cv2.putText(frame, f"EAR:{features.mean_ear:.2f} {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("EyeCaptureModel Usage Example", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

    summary = model.get_session_features()
    print(json.dumps(summary.to_dict(), indent=2))

    model.close()


if __name__ == "__main__":
    main()