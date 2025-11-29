#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate a trained XGBoost model on nuScenes meta CSV.

Loads:
  /home/frank/Pu/sci_ML/Tools_nuscenes/xgb_nuscenes_bestmodel_XGBoost_v2.bin

Evaluates on:
  /home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv
"""

import ast
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb


# ============================== PATHS ==============================
MODEL_PATH = "/home/frank/Pu/sci_ML/Tools_nuscenes/xgb_nuscenes_bestmodel_XGBoost_v2.bin"
CSV_PATH   = "/home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv"

# ============================ CLASS INFO ============================
NUSC_CLASSES = [
    "car", "truck", "bus", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle",
    "traffic_cone", "barrier", "trailer"
]

MODEL_TO_ID = {
    "bevdal": 0,
    "bevfusion": 1,
    "transfusion": 2,
}

ID_TO_MODEL = {v: k for k, v in MODEL_TO_ID.items()}


# ================== Feature Builder (same as training) ==================

def build_features_from_row(row: pd.Series) -> Tuple[np.ndarray, int]:
    names = ast.literal_eval(row["detection_names"])
    scores = ast.literal_eval(row["detection_scores"])
    assert len(names) == len(scores)

    scores_arr = np.asarray(scores, dtype=np.float32)

    num_det = float(row["num_detections"])
    overall_max = float(scores_arr.max()) if len(scores_arr) > 0 else 0.0
    overall_mean = float(scores_arr.mean()) if len(scores_arr) > 0 else 0.0
    overall_std = float(scores_arr.std()) if len(scores_arr) > 0 else 0.0

    best_nd_score = float(row.get("best_nd_score", 0.0))
    max_det_score = float(row.get("max_det_score", overall_max))
    rank_in_model = float(row.get("rank_in_model", 0.0))

    class_counts: List[float] = []
    class_max_scores: List[float] = []

    for cls in NUSC_CLASSES:
        cls_scores = [s for n, s in zip(names, scores) if n == cls]
        if len(cls_scores) == 0:
            class_counts.append(0.0)
            class_max_scores.append(0.0)
        else:
            class_counts.append(len(cls_scores))
            class_max_scores.append(max(cls_scores))

    feat = np.array(
        [num_det, overall_max, overall_mean, overall_std,
         best_nd_score, max_det_score, rank_in_model]
        + class_counts + class_max_scores,
        dtype=np.float32,
    )

    # parse label
    bm_raw = str(row["best_model"]).strip()
    bm_key = bm_raw.lower()

    if bm_key not in MODEL_TO_ID:
        raise ValueError(f"Unknown best_model {bm_raw} for sample_token {row['sample_token']}")

    label = MODEL_TO_ID[bm_key]
    return feat, label


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"[Info] Loaded CSV with {len(df)} rows.")

    X_list = []
    y_list = []
    sample_tokens = []

    for i, row in df.iterrows():
        feat, lab = build_features_from_row(row)
        X_list.append(feat)
        y_list.append(lab)
        sample_tokens.append(row["sample_token"])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print(f"[Info] Built dataset: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y, sample_tokens


# =============================== MAIN ==================================

def main():
    print(f"[Load Model] {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    X, y_true, sample_tokens = load_dataset(CSV_PATH)

    # Predict
    y_pred = model.predict(X)

    # ===== Metrics =====
    acc = accuracy_score(y_true, y_pred)
    print("\n==================== Validation Result ====================")
    print(f"Accuracy: {acc:.4f}\n")

    print("[Classification Report]")
    print(classification_report(
        y_true, y_pred,
        target_names=[ID_TO_MODEL[i] for i in range(len(ID_TO_MODEL))]
    ))

    print("[Confusion Matrix]")
    print(confusion_matrix(y_true, y_pred))

    # ====== Save per-sample predictions ======
    out_csv = "/home/frank/Pu/sci_ML/Tools_nuscenes/validation_predictions.csv"
    df_out = pd.DataFrame({
        "sample_token": sample_tokens,
        "true_label": [ID_TO_MODEL[i] for i in y_true],
        "pred_label": [ID_TO_MODEL[i] for i in y_pred]
    })
    df_out.to_csv(out_csv, index=False)
    print(f"\n[Saved] Detailed results written to: {out_csv}")


if __name__ == "__main__":
    main()
