import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ModelEyeGaze:

    def __init__(self,
                 model_path="face_landmarker.task",
                 classifier_path="eye_gaze.pkl"):

        print("Loading MediaPipe FaceLandmarker...")

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.classifier = joblib.load(classifier_path)

        print("Dual Eye Gaze Model Ready!")

    def process(self, frame):

        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return frame

        landmarks = result.face_landmarks[0]

        # RIGHT EYE
        R_LEFT, R_RIGHT = 33, 133
        R_TOP, R_BOTTOM = 159, 145
        R_IRIS = 468

        # LEFT EYE
        L_LEFT, L_RIGHT = 362, 263
        L_TOP, L_BOTTOM = 386, 374
        L_IRIS = 473

        def to_pixel(lm):
            return int(lm.x * img_w), int(lm.y * img_h)

        # Right eye pixels
        r_left_x, r_left_y = to_pixel(landmarks[R_LEFT])
        r_right_x, r_right_y = to_pixel(landmarks[R_RIGHT])
        r_top_x, r_top_y = to_pixel(landmarks[R_TOP])
        r_bottom_x, r_bottom_y = to_pixel(landmarks[R_BOTTOM])
        r_iris_x, r_iris_y = to_pixel(landmarks[R_IRIS])

        # Left eye pixels
        l_left_x, l_left_y = to_pixel(landmarks[L_LEFT])
        l_right_x, l_right_y = to_pixel(landmarks[L_RIGHT])
        l_top_x, l_top_y = to_pixel(landmarks[L_TOP])
        l_bottom_x, l_bottom_y = to_pixel(landmarks[L_BOTTOM])
        l_iris_x, l_iris_y = to_pixel(landmarks[L_IRIS])

        # Draw
        cv2.circle(frame, (r_iris_x, r_iris_y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (l_iris_x, l_iris_y), 3, (0, 255, 0), -1)

        # Compute dimensions
        r_width = np.linalg.norm([r_right_x - r_left_x,
                                  r_right_y - r_left_y])
        r_height = np.linalg.norm([r_bottom_x - r_top_x,
                                   r_bottom_y - r_top_y])

        l_width = np.linalg.norm([l_right_x - l_left_x,
                                  l_right_y - l_left_y])
        l_height = np.linalg.norm([l_bottom_x - l_top_x,
                                   l_bottom_y - l_top_y])

        if r_width == 0 or r_height == 0 or l_width == 0 or l_height == 0:
            return frame

        # Ratios
        r_h = (r_iris_x - r_left_x) / r_width
        r_v = (r_iris_y - r_top_y) / r_height

        l_h = (l_iris_x - l_left_x) / l_width
        l_v = (l_iris_y - l_top_y) / l_height

        # Clip
        r_h, r_v = np.clip([r_h, r_v], 0, 1)
        l_h, l_v = np.clip([l_h, l_v], 0, 1)

        # Prepare DataFrame (IMPORTANT to avoid warning)
        features = pd.DataFrame([[
            r_h, r_v, l_h, l_v
        ]], columns=[
            "r_h",
            "r_v",
            "l_h",
            "l_v"
        ])

        prediction = self.classifier.predict(features)[0]

        cv2.putText(frame,
                    f"Gaze: {prediction}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        return frame