#!/usr/bin/env python3
import os
import cv2
import numpy as np
from mtcnn import MTCNN
import onnxruntime as ort
import argparse

# ---------- Defaults ----------
DEFAULT_MODEL = os.path.join("models", "arcface.onnx")
DEFAULT_DB_FILE = "db_embeddings.npz"
FACE_SIZE = (112, 112)
THRESHOLD = 0.45

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

# ---------- ArcFace ----------
class ArcFace:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print("Loaded ArcFace:", model_path)

    def get_embedding(self, face_bgr):
        x = preprocess_face(face_bgr)
        emb = self.session.run([self.output_name], {self.input_name: x})[0]
        emb = emb.reshape(-1)
        emb = l2_norm(emb)
        return emb

# ---------- Load DB ----------
def load_db(db_file):
    if not os.path.exists(db_file):
        print(f"[realtime] Embedding file not found: {db_file}")
        return {}
    data = np.load(db_file, allow_pickle=True)
    names = data.get("names", None)
    embs = data.get("embs", None)
    if names is None or embs is None or len(names) != embs.shape[0]:
        print(f"[realtime] Invalid or empty DB file: {db_file}")
        return {}
    # ensure names are Python strings
    names = [str(n) for n in names]
    db = {n: embs[i] for i, n in enumerate(names)}
    print(f"[realtime] Loaded DB with {len(db)} identities from {db_file}")
    return db

# ---------- Recognition ----------
def recognize(embedding, db, threshold=THRESHOLD):
    best_name = "Unknown"
    best_score = -1
    for name, db_emb in db.items():
        score = float(np.dot(embedding, db_emb))
        if score > best_score:
            best_score = score
            best_name = name
    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Real-time face recognition using precomputed embeddings")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to arcface.onnx")
    p.add_argument("--dbfile", default=DEFAULT_DB_FILE, help="Embeddings file produced by train_database.py (.npz)")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="Similarity threshold (dot product)")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    return p.parse_args()

def main():
    args = parse_args()
    arcface = ArcFace(args.model)
    detector = MTCNN()
    db = load_db(args.dbfile)

    if not db:
        print("[realtime] Database empty. Run train_database.py first or provide a valid .npz file.")
        # still continue to show camera, but everything will be Unknown

    cap = cv2.VideoCapture(args.camera)
    print("[realtime] Starting webcam. Press 'q' to quit.")
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
                name, score = recognize(emb, db, threshold=args.threshold)
            except Exception:
                name, score = "Unknown", -1.0

            label = f"{name} | {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
