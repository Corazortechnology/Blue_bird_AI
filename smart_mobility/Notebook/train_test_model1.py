import os
import cv2
import random
import shutil
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from models.model2 import ArcFaceModel
from retinaface import RetinaFace


DATASET_PATH = "/home/mluser/data/smart_mobility_harshit/Smart_mobility_ai/smart_mobility/Notebook/database/train"
TRAIN_DIR = "temp_train"
TEST_DIR = "temp_test"

PREDICTION_CSV = "arcface_predictions.csv"
METRICS_CSV = "arcface_metrics.csv"


# ---------------------------------------------------
# Split dataset per person (80/20)
# ---------------------------------------------------
def split_dataset():

    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    os.makedirs(TRAIN_DIR)
    os.makedirs(TEST_DIR)

    for person in os.listdir(DATASET_PATH):

        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        images = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if len(images) == 0:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * 0.8)

        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        os.makedirs(os.path.join(TRAIN_DIR, person))
        os.makedirs(os.path.join(TEST_DIR, person))

        for img in train_imgs:
            shutil.copy(
                os.path.join(person_path, img),
                os.path.join(TRAIN_DIR, person, img)
            )

        for img in test_imgs:
            shutil.copy(
                os.path.join(person_path, img),
                os.path.join(TEST_DIR, person, img)
            )

    print("✅ Dataset split complete (80% train / 20% test)")


# ---------------------------------------------------
# Train ArcFace (build embedding DB)
# ---------------------------------------------------
def train_model():

    print("\n🔵 Initializing ArcFace model...")
    model = ArcFaceModel()

    print("🔨 Building embedding database...")
    model.build_database_in_chunks(
        TRAIN_DIR,
        out_path="models/arcface_db.npz",
        chunk_size=1000,
        verbose=True
    )

    print("✅ Training complete.")

    return ArcFaceModel(load_db=True, db_path="models/arcface_db.npz")


# ---------------------------------------------------
# Evaluate model
# ---------------------------------------------------
def evaluate(model):

    y_true = []
    y_pred = []
    image_names = []

    print("\n🔍 Running evaluation...")

    persons = [
        p for p in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, p))
    ]

    for person in persons:

        img_paths = glob(os.path.join(TEST_DIR, person, "*"))

        for img_path in img_paths:

            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                detections = RetinaFace.detect_faces(img)
            except:
                continue

            if not detections:
                continue

            faces = list(detections.values())
            largest = max(
                faces,
                key=lambda d: (d["facial_area"][2] - d["facial_area"][0]) *
                              (d["facial_area"][3] - d["facial_area"][1])
            )

            x1, y1, x2, y2 = largest["facial_area"]
            face = img[y1:y2, x1:x2]

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            emb = model.get_embedding(face)
            pred_name, score = model.recognize(emb)

            y_true.append(person)
            y_pred.append(pred_name)
            image_names.append(os.path.basename(img_path))

    if len(y_true) == 0:
        print("❌ No valid test samples.")
        return

    # ---------------- Metrics ----------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    print("\n========= TEST RESULTS =========")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # ---------------- Save Prediction CSV ----------------
    prediction_df = pd.DataFrame({
        "image_name": image_names,
        "actual_label": y_true,
        "predicted_label": y_pred
    })
    prediction_df.to_csv(PREDICTION_CSV, index=False)

    # ---------------- Save Metrics CSV ----------------
    metrics_df = pd.DataFrame({
        "accuracy": [accuracy],
        "precision_macro": [precision],
        "recall_macro": [recall],
        "f1_macro": [f1]
    })
    metrics_df.to_csv(METRICS_CSV, index=False)

    print(f"\nSaved predictions to: {PREDICTION_CSV}")
    print(f"Saved metrics to: {METRICS_CSV}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    random.seed(42)

    print("Loading dataset...")
    split_dataset()

    model = train_model()

    evaluate(model)