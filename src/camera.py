"""
camera.py

Responsibility: ONLY webcam frame acquisition.
No face detection, no feature extraction, no drawing logic belongs here.
Other modules will import CameraStream and call its methods.
"""

import cv2
import time


class CameraStream:
    """
    Wraps OpenCV's VideoCapture to provide a simple, reusable interface
    for reading frames from a webcam, with basic error handling and FPS tracking.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Args:
            camera_index: which camera to use (0 = default/first webcam).
            width, height: requested capture resolution. The camera may not
                honor these exactly — we verify the actual resolution after opening.
        """
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.cap = None
        self._last_frame_time = None
        self.fps = 0.0

    def open(self) -> bool:
        """
        Opens the camera device. Returns True on success, False on failure.
        """
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW = faster startup on Windows

        if not self.cap.isOpened():
            print(f"[camera.py] ERROR: Could not open camera index {self.camera_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[camera.py] Camera opened. Requested {self.requested_width}x{self.requested_height}, "
              f"actual {actual_width}x{actual_height}")

        return True

    def read_frame(self):
        """
        Reads a single frame from the camera.

        Returns:
            (success: bool, frame: np.ndarray or None, timestamp: float)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, time.time()

        success, frame = self.cap.read()
        timestamp = time.time()

        # Track instantaneous FPS based on time between successful reads
        if success and self._last_frame_time is not None:
            delta = timestamp - self._last_frame_time
            if delta > 0:
                self.fps = 1.0 / delta
        self._last_frame_time = timestamp

        return success, frame, timestamp

    def release(self):
        """
        Releases the camera device. Must be called when done, or the camera
        can remain locked (other apps won't be able to use it) until the
        process exits.
        """
        if self.cap is not None:
            self.cap.release()
            print("[camera.py] Camera released.")


def _standalone_test():
    """
    Simple standalone test: opens the webcam, shows the live feed,
    displays current FPS, and exits when 'q' is pressed.
    Run this file directly (python src/camera.py) to test camera capture alone.
    """
    stream = CameraStream(camera_index=0)

    if not stream.open():
        print("Exiting: camera failed to open.")
        return

    print("Press 'q' in the video window to quit.")

    while True:
        success, frame, timestamp = stream.read_frame()

        if not success:
            print("[camera.py] WARNING: Failed to read frame. Retrying...")
            continue

        # Overlay FPS on the frame for a quick sanity check
        cv2.putText(
            frame,
            f"FPS: {stream.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Stage 1 - Webcam Capture Test", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _standalone_test()