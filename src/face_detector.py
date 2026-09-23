"""
face_detector.py

Responsibility: ONLY face + landmark detection via MediaPipe's FaceLandmarker.
Wraps the Tasks API so the rest of the codebase never touches mediapipe
internals directly — if the API changes again, only this file needs updating.
"""

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np
from typing import Optional, List, Tuple


class FaceDetector:
    """
    Wraps MediaPipe FaceLandmarker for synchronous, per-frame use in a
    webcam loop (VIDEO running mode — requires monotonically increasing
    timestamps, which we generate from a simple frame counter).
    """

    def __init__(self, model_path: str = "models/face_landmarker.task",
                 num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,             # we don't need expressions
            output_facial_transformation_matrixes=False # we don't need 3D pose matrices
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_index = 0  # used to generate increasing timestamps for VIDEO mode

    def detect(self, frame_bgr: np.ndarray) -> Optional[List]:
        """
        Runs face landmark detection on a single BGR frame (as read by OpenCV).

        Returns:
            A list of landmarks for the first detected face (list of objects
            with .x, .y, .z in normalized [0,1] image coordinates), or None
            if no face was detected.
        """
        # MediaPipe expects RGB, OpenCV gives BGR — convert.
        frame_rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode requires an increasing timestamp in milliseconds.
        timestamp_ms = int(self._frame_index * (1000 / 30))  # assume ~30fps spacing
        self._frame_index += 1

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]  # first detected face only (num_faces=1)

    @staticmethod
    def get_bounding_box(landmarks, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """
        Computes a pixel-space bounding box (x_min, y_min, x_max, y_max)
        from normalized landmark coordinates. Used for Stage 2 visualization
        only — later stages will use specific landmark indices instead.
        """
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x_min = int(min(xs) * frame_width)
        x_max = int(max(xs) * frame_width)
        y_min = int(min(ys) * frame_height)
        y_max = int(max(ys) * frame_height)

        return x_min, y_min, x_max, y_max

    def close(self):
        """Releases the MediaPipe landmarker. Call when done."""
        self._landmarker.close()


def _standalone_test():
    """
    Combines CameraStream + FaceDetector to visually confirm detection works.
    Run: python src/face_detector.py
    """
    import cv2
    from camera import CameraStream  # same folder, run from src/ context

    stream = CameraStream(camera_index=0)
    if not stream.open():
        print("Exiting: camera failed to open.")
        return

    detector = FaceDetector(model_path="../models/face_landmarker.task")
    print("Press 'q' to quit.")

    while True:
        success, frame, _ = stream.read_frame()
        if not success:
            continue

        landmarks = detector.detect(frame)
        h, w = frame.shape[:2]

        if landmarks is not None:
            x_min, y_min, x_max, y_max = detector.get_bounding_box(landmarks, w, h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected", (x_min, max(y_min - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stage 2 - Face Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _standalone_test()