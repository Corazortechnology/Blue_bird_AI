import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 2
PERCLOS_WINDOW = 30  # seconds window
DROWSY_THRESHOLD = 0.4

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def compute_ear(eye):
    p1, p2, p3, p4, p5, p6 = eye
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


class Model3Drowsiness:

    def __init__(self, model_path: str = "models/face_landmarker.task"):
        print("Loading MediaPipe FaceLandmarker...")

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.blink_count = 0
        self.frame_counter = 0
        self.closed_frames = deque()
        self.perclos = 0.0

        print("Model3 ready.")

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        current_time = time.time()
        eyes_closed = False

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            h, w, _ = frame.shape

            left_eye = np.array(
                [[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE]
            )
            right_eye = np.array(
                [[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE]
            )

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)

            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                self.frame_counter += 1
                eyes_closed = True
            else:
                if self.frame_counter >= CONSEC_FRAMES:
                    self.blink_count += 1
                self.frame_counter = 0

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Update PERCLOS window
        self.closed_frames.append((current_time, eyes_closed))

        while self.closed_frames and (
            current_time - self.closed_frames[0][0] > PERCLOS_WINDOW
        ):
            self.closed_frames.popleft()

        if self.closed_frames:
            closed_count = sum(1 for t, closed in self.closed_frames if closed)
            self.perclos = closed_count / len(self.closed_frames)

        # Drowsiness check
        if self.perclos > DROWSY_THRESHOLD:
            cv2.putText(
                frame,
                "DROWSY!",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

        cv2.putText(
            frame,
            f"PERCLOS: {self.perclos:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Blinks: {self.blink_count}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return frame