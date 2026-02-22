import cv2
import numpy as np
import mediapipe as mp
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ModelHeadPose:

    def __init__(self,
                 model_path="face_landmarker.task",
                 classifier_path="headpose_classifier.pkl"):

        print("Loading MediaPipe FaceLandmarker...")

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.classifier = joblib.load(classifier_path)

        print("HeadPose model ready.")

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

        face_2d = []
        face_3d = []
        landmark_ids = [33, 263, 1, 61, 291, 199]

        for idx in landmark_ids:
            lm = landmarks[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w

        cam_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d,
            face_2d,
            cam_matrix,
            dist_matrix,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return frame

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360

        prediction = self.classifier.predict([[pitch, yaw, roll]])[0]

        cv2.putText(frame, f"Direction: {prediction}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

        return frame