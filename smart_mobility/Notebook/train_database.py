#!/usr/bin/env python3
import os
import cv2
import numpy as np
from mtcnn import MTCNN
import onnxruntime as ort
from glob import glob
import argparse

# ---------- Defaults (change with CLI) ----------
DEFAULT_MODEL = os.path.join("models", "arcface.onnx")
DEFAULT_DATABASE = "database"
DEFAULT_OUT = "db_embeddings.npz"
FACE_SIZE = (112, 112)
THRESHOLD = 0.45  # not used here, but kept for consistency

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
            raise FileNotFoundError(f"ArcFace ONNX model not found: {model_path}")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print("Loaded ArcFace:", model_path)
        print("Model Input Shape:", self.session.get_inputs()[0].shape)

    def get_embedding(self, face_bgr):
        x = preprocess_face(face_bgr)
        emb = self.session.run([self.output_name], {self.input_name: x})[0]
        emb = emb.reshape(-1)
        emb = l2_norm(emb)
        return emb

# ---------- Build Raw Image DB ----------
def build_raw_database(database_dir):
    raw_db = {}

    if not os.path.exists(database_dir):
        print(f"[train] Database dir not found: {database_dir}")
        return raw_db

    persons = [p for p in os.listdir(database_dir)
               if os.path.isdir(os.path.join(database_dir, p))]

    for person in persons:
        images = glob(os.path.join(database_dir, person, "*"))
        img_list = []

        for path in images:
            img = cv2.imread(path)
            if img is not None:
                img_list.append(img)

        if img_list:
            raw_db[person] = img_list
        else:
            print(f"[train] No readable images for {person}, skipping")

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
            largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
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
            print(f"[train] Added {name} with {len(embeddings)} samples")
        else:
            print(f"[train] Skipped {name} (no valid faces/embeddings)")

    return db

# ---------- Save embeddings ----------
def save_db(db, out_path):
    if not db:
        print("[train] No embeddings to save.")
        # still create empty file
        np.savez_compressed(out_path, names=np.array([]), embs=np.array([]))
        print(f"[train] Saved empty DB to {out_path}")
        return

    names = np.array(list(db.keys()))
    embs = np.stack(list(db.values()), axis=0)  # shape (N, D)
    np.savez_compressed(out_path, names=names, embs=embs)
    print(f"[train] Saved DB with {len(names)} identities to {out_path}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train / build ArcFace DB embeddings")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to arcface.onnx")
    p.add_argument("--db", default=DEFAULT_DATABASE, help="Database directory (person subfolders)")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output embeddings file (.npz)")
    return p.parse_args()

def main():
    args = parse_args()
    arcface = ArcFace(args.model)
    detector = MTCNN()
    raw_db = build_raw_database(args.db)
    if not raw_db:
        print("[train] No persons found in database. Exiting.")
    db = compute_database_embeddings(arcface, detector, raw_db)
    save_db(db, args.out)

if __name__ == "__main__":
    main()
