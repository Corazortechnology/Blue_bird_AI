import os
import cv2
import numpy as np
from mtcnn import MTCNN
import onnxruntime as ort
from glob import glob
import time

# ---------- Config ----------
MODEL_DIR = "models"
ARCFACE_ONNX = os.path.join(MODEL_DIR, "arcface.onnx")

DATABASE_DIR = "database"
THRESHOLD = 0.45
FACE_SIZE = (112, 112)

# ---------- Utility ----------
def l2_norm(x, epsilon=1e-10):
    return x / (np.linalg.norm(x) + epsilon)


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, FACE_SIZE)
    arr = face_resized.astype(np.float32)
    arr = (arr - 127.5) / 128.0
    arr = np.expand_dims(arr, axis=0)  # NHWC
    return arr


# ---------- ArcFace Wrapper ----------
class ArcFace:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("ArcFace ONNX model not found.")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("Model Input Shape:",
              self.session.get_inputs()[0].shape)

    def get_embedding(self, face_bgr):
        x = preprocess_face(face_bgr)
        emb = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )[0]
        emb = emb.reshape(-1)
        emb = l2_norm(emb)
        return emb


# ---------- Build Raw Image DB ----------
def build_raw_database():
    raw_db = {}

    if not os.path.exists(DATABASE_DIR):
        return raw_db

    persons = [p for p in os.listdir(DATABASE_DIR)
               if os.path.isdir(os.path.join(DATABASE_DIR, p))]

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


# ---------- Compute Embeddings ----------
def compute_database_embeddings(arcface, detector, raw_db):
    db = {}

    for name, images in raw_db.items():
        embeddings = []

        for img in images:

            if img is None:
                continue

            if img.shape[0] < 100 or img.shape[1] < 100:
                continue

            try:
                detections = detector.detect_faces(img)
            except Exception:
                continue

            if not detections:
                continue

            # pick largest face
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

            try:
                emb = arcface.get_embedding(face)
                embeddings.append(emb)
            except Exception:
                continue

        if embeddings:
            avg_emb = np.mean(np.stack(embeddings), axis=0)
            avg_emb = l2_norm(avg_emb)
            db[name] = avg_emb
            print(f"[DB] Added {name} with {len(embeddings)} samples")
        else:
            print(f"[DB] Skipped {name}")

    return db


# ---------- Recognition ----------
def recognize(embedding, db):
    best_name = "Unknown"
    best_score = -1

    for name, db_emb in db.items():
        score = np.dot(embedding, db_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


# ---------- MAIN ----------
def main():
    print("Starting Webcam Face Recognition (CPU Mode)")

    arcface = ArcFace(ARCFACE_ONNX)
    detector = MTCNN()

    raw_db = build_raw_database()
    db = compute_database_embeddings(arcface, detector, raw_db)

    if not db:
        print("No database embeddings found.")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            detections = detector.detect_faces(frame)
        except Exception:
            detections = []

        for d in detections:
            x, y, w, h = d['box']

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            face = frame[y1:y2, x1:x2]

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            try:
                emb = arcface.get_embedding(face)
                name, score = recognize(emb, db)
            except Exception:
                continue

            label = f"{name} | {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 0, 0), 2)

            cv2.putText(frame, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
