import os
import cv2
import numpy as np
from mtcnn import MTCNN
import onnxruntime as ort
from glob import glob

MODEL_DIR = "models"
ARCFACE_ONNX = os.path.join(MODEL_DIR, "arcface.onnx")
DATABASE_DIR = "models/database"

THRESHOLD = 0.45
FACE_SIZE = (112, 112)


def l2_norm(x, epsilon=1e-10):
    return x / (np.linalg.norm(x) + epsilon)


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, FACE_SIZE)
    arr = face_resized.astype(np.float32)
    arr = (arr - 127.5) / 128.0
    arr = np.expand_dims(arr, axis=0)
    return arr


class Model2ArcFace:

    def __init__(self):
        print("🔵 Loading ArcFace Model...")
        print("Looking for database at:", os.path.abspath(DATABASE_DIR))

        self.detector = MTCNN()

        self.session = ort.InferenceSession(
            ARCFACE_ONNX,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("✅ ArcFace Model Loaded")

        print("🔵 Building Face Database...")
        raw_db = self.build_raw_database()
        self.db = self.compute_database_embeddings(raw_db)

        print("✅ Database Ready")
        print("=====================================")

    # --------------------------------------
    # DATABASE BUILDING
    # --------------------------------------

    def build_raw_database(self):
        raw_db = {}

        if not os.path.exists(DATABASE_DIR):
            print("❌ Database folder not found")
            return raw_db

        persons = [
            p for p in os.listdir(DATABASE_DIR)
            if os.path.isdir(os.path.join(DATABASE_DIR, p))
        ]

        for person in persons:
            images = glob(os.path.join(DATABASE_DIR, person, "*"))
            img_list = []

            for path in images:
                img = cv2.imread(path)
                if img is not None:
                    img_list.append(img)

            if img_list:
                raw_db[person] = img_list

        return raw_db

    def compute_database_embeddings(self, raw_db):
        db = {}

        for name, images in raw_db.items():
            embeddings = []

            for img in images:
                try:
                    detections = self.detector.detect_faces(img)
                except Exception:
                    continue

                if not detections:
                    continue

                largest = max(
                    detections,
                    key=lambda d: d['box'][2] * d['box'][3]
                )

                x, y, w, h = largest['box']

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img.shape[1], x + w)
                y2 = min(img.shape[0], y + h)

                face = img[y1:y2, x1:x2]

                if face.shape[0] < 50 or face.shape[1] < 50:
                    continue

                emb = self.get_embedding(face)
                embeddings.append(emb)

            if embeddings:
                avg_emb = np.mean(np.stack(embeddings), axis=0)
                avg_emb = l2_norm(avg_emb)
                db[name] = avg_emb
                print(f"[TRAINED] {name} → {len(embeddings)} samples")
            else:
                print(f"[SKIPPED] {name}")

        return db

    # --------------------------------------
    # EMBEDDING
    # --------------------------------------

    def get_embedding(self, face):
        x = preprocess_face(face)

        emb = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )[0]

        emb = emb.reshape(-1)
        emb = l2_norm(emb)

        return emb

    # --------------------------------------
    # RECOGNITION
    # --------------------------------------

    def recognize(self, embedding):
        best_name = "Unknown"
        best_score = -1

        for name, db_emb in self.db.items():
            score = np.dot(embedding, db_emb)

            if score > best_score:
                best_score = score
                best_name = name

        if best_score < THRESHOLD:
            return "Unknown", best_score

        return best_name, best_score

    # --------------------------------------
    # FRAME PROCESSING (WebSocket)
    # --------------------------------------

    def process(self, frame):

        try:
            detections = self.detector.detect_faces(frame)
        except Exception:
            return frame

        for d in detections:
            x, y, w, h = d['box']

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            face = frame[y1:y2, x1:x2]

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            emb = self.get_embedding(face)
            name, score = self.recognize(emb)

            label = f"{name} | {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame
