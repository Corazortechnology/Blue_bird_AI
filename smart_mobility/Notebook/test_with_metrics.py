#!/usr/bin/env python3
import os
import cv2
import numpy as np
from mtcnn import MTCNN
import onnxruntime as ort
from glob import glob
import argparse
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Defaults ----------
DEFAULT_MODEL = os.path.join("models", "arcface.onnx")
DEFAULT_DB_FILE = "db_embeddings.npz"
DEFAULT_TEST_DIR = "database"
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
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------- ArcFace ----------
class ArcFace:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print("✓ Loaded ArcFace model")

    def get_embedding(self, face_bgr):
        x = preprocess_face(face_bgr)
        emb = self.session.run([self.output_name], {self.input_name: x})[0]
        emb = emb.reshape(-1)
        emb = l2_norm(emb)
        return emb


# ---------- Load Database ----------
def load_db(db_file):
    if not os.path.exists(db_file):
        raise FileNotFoundError("Embedding file not found. Run train_database.py first.")

    data = np.load(db_file, allow_pickle=True)
    names = data["names"]
    embs = data["embs"]

    db = {str(names[i]): embs[i] for i in range(len(names))}
    print(f"✓ Loaded {len(db)} identities from DB")
    return db


# ---------- Recognition ----------
def recognize(embedding, db, threshold):
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


# ---------- Test Images ----------
def test_images(arcface, detector, db, test_dir, threshold):

    y_true = []
    y_pred = []

    persons = [p for p in os.listdir(test_dir)
               if os.path.isdir(os.path.join(test_dir, p))]

    print("\n===== Testing Started =====\n")

    for person in persons:
        image_paths = glob(os.path.join(test_dir, person, "*"))

        for img_path in image_paths:
            img = cv2.imread(img_path)

            if img is None:
                continue

            if img.shape[0] < 100 or img.shape[1] < 100:
                continue

            try:
                detections = detector.detect_faces(img)
            except:
                continue

            if not detections:
                continue

            largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
            x, y, w, h = largest['box']

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            face = img[y1:y2, x1:x2]

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            try:
                emb = arcface.get_embedding(face)
                pred_name, score = recognize(emb, db, threshold)
            except:
                continue

            print(f"Actual: {person} | Predicted: {pred_name} | Score: {score:.3f}")

            y_true.append(person)
            y_pred.append(pred_name)

    return y_true, y_pred


# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Test ArcFace Model with Metrics")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dbfile", default=DEFAULT_DB_FILE)
    parser.add_argument("--testdir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


# ---------- Main ----------
def main():
    args = parse_args()

    arcface = ArcFace(args.model)
    detector = MTCNN()
    db = load_db(args.dbfile)

    y_true, y_pred = test_images(
        arcface,
        detector,
        db,
        args.testdir,
        args.threshold
    )

    if len(y_true) == 0:
        print("No valid test samples found.")
        return

    print("\n========== FINAL RESULTS ==========\n")

    # -------- Accuracy --------
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")

    # -------- Classification Report --------
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    # -------- Confusion Matrix --------
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Save metrics CSV
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    metrics_df = pd.DataFrame({
        "Class": labels,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support
    })

    metrics_df.loc[len(metrics_df)] = ["OVERALL_ACCURACY", acc, "", "", ""]
    metrics_df.to_csv("results_metrics.csv", index=False)

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv("confusion_matrix.csv")

    print("✓ Saved results_metrics.csv")
    print("✓ Saved confusion_matrix.csv")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel("Predicted") 
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
