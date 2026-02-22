import argparse
import os
import random
import shutil
import cv2
from glob import glob
import numpy as np
import pandas as pd
from models.model2 import ArcFaceModel
from retinaface import RetinaFace


# -------------------------------------------------------
# 1️⃣ Split Dataset (80 / 20 per person)
# -------------------------------------------------------
def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir)
    os.makedirs(test_dir)

    for person in os.listdir(source_dir):

        person_path = os.path.join(source_dir, person)
        if not os.path.isdir(person_path):
            continue

        images = [f for f in os.listdir(person_path)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        if len(images) == 0:
            continue

        random.shuffle(images)
        split_index = int(len(images) * split_ratio)

        train_imgs = images[:split_index]
        test_imgs = images[split_index:]

        os.makedirs(os.path.join(train_dir, person))
        os.makedirs(os.path.join(test_dir, person))

        for img in train_imgs:
            shutil.copy(os.path.join(person_path, img),
                        os.path.join(train_dir, person, img))

        for img in test_imgs:
            shutil.copy(os.path.join(person_path, img),
                        os.path.join(test_dir, person, img))

    print("✅ Dataset split complete (80% train / 20% test)")


# -------------------------------------------------------
# 2️⃣ Evaluation
# -------------------------------------------------------
def evaluate(model: ArcFaceModel, test_dir: str, output_csv: str):

    persons = sorted([
        p for p in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, p))
    ])

    y_true = []
    y_pred = []

    print("\n🔍 Running Evaluation...\n")

    for person in persons:
        img_paths = glob(os.path.join(test_dir, person, "*"))

        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                detections = RetinaFace.detect_faces(img)
            except Exception:
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
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            face = img[y1:y2, x1:x2]
            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            emb = model.get_embedding(face)
            pred_name, score = model.recognize(emb)

            y_true.append(person)
            y_pred.append(pred_name)

    if len(y_true) == 0:
        print("❌ No valid faces found in test set.")
        return

    total = len(y_true)
    correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])
    accuracy = correct / total

    print(f"\nTotal Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    labels = sorted(list(set(y_true + y_pred)))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    metrics = []

    for i, label in enumerate(labels):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics.append({
            "Class": label,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df["Overall Accuracy"] = accuracy
    metrics_df.to_csv(output_csv, index=False)

    print(f"\n✅ Metrics saved to: {output_csv}")


# -------------------------------------------------------
# 3️⃣ MAIN
# -------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to labeled folders (person subfolders)")
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--out_db", type=str, default="models/arcface_db.npz")
    parser.add_argument("--out_csv", type=str, default="evaluation_results.csv")

    args = parser.parse_args()

    random.seed(42)

    train_dir = "temp_train"
    test_dir = "temp_test"

    # 1️⃣ Split
    split_dataset(args.data, train_dir, test_dir)

    # 2️⃣ Train
    print("\n🔵 Initializing ArcFace model...")
    model = ArcFaceModel()

    print("🔨 Building database from training set...")
    model.build_database_in_chunks(
        train_dir,
        out_path=args.out_db,
        chunk_size=args.chunk,
        verbose=True
    )

    print("✅ Training complete.")

    # 3️⃣ Load DB
    model = ArcFaceModel(load_db=True, db_path=args.out_db)

    # 4️⃣ Test
    evaluate(model, test_dir, args.out_csv)
