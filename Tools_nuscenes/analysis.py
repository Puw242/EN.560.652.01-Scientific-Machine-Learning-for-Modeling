#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation script for the trained XGBoost model:

  /home/frank/Pu/sci_ML/Tools_nuscenes/xgb_nuscenes_bestmodel_XGBoost_v2.bin

It rebuilds features from the nuScenes meta CSV and evaluates:
  - overall accuracy
  - classification report
  - confusion matrix
  - per-class accuracy
  - optional: per-sample prediction dump
"""

import ast
import numpy as np
import pandas as pd

from typing import Tuple, List

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb

# ====================== CONFIG ======================

MODEL_PATH = "/home/frank/Pu/sci_ML/Tools_nuscenes/xgb_nuscenes_bestmodel_XGBoost_v2.bin"

# 按需修改：验证集 CSV
CSV_PATH = "/home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv"

# nuScenes 中的主要类别（要和训练时保持一致）
NUSC_CLASSES = [
    "car",
    "truck",
    "bus",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
    "trailer",
]

# 模型标签映射（保持和训练脚本一致）
MODEL_TO_ID = {
    "bevdal": 0,
    "bevfusion": 1,
    "transfusion": 2,
}
ID_TO_MODEL = {v: k for k, v in MODEL_TO_ID.items()}


# ====================== FEATURE BUILDING ======================

def build_features_from_row(row: pd.Series) -> Tuple[np.ndarray, int, str]:
    """
    从一行 CSV 构造特征向量 X 和标签 y 以及 sample_token。

    特征结构:
      [ num_detections,
        overall_max_score, overall_mean_score, overall_std_score,
        best_nd_score, max_det_score, rank_in_model,
        counts for each class (10),
        max score for each class (10)
      ]
    """
    # ---------- 解析 detection_names / detection_scores ----------
    names = ast.literal_eval(row["detection_names"])
    scores = ast.literal_eval(row["detection_scores"])

    assert len(names) == len(scores), "names 和 scores 长度不一致"

    scores_arr = np.asarray(scores, dtype=np.float32)

    num_det = float(row["num_detections"])
    overall_max = float(scores_arr.max()) if len(scores_arr) > 0 else 0.0
    overall_mean = float(scores_arr.mean()) if len(scores_arr) > 0 else 0.0
    overall_std = float(scores_arr.std()) if len(scores_arr) > 0 else 0.0

    best_nd_score = float(row.get("best_nd_score", 0.0))
    max_det_score = float(row.get("max_det_score", overall_max))
    rank_in_model = float(row.get("rank_in_model", 0.0))

    # ---------- 每个类别的 count & max score ----------
    class_counts: List[float] = []
    class_max_scores: List[float] = []

    for cls in NUSC_CLASSES:
        cls_scores = [s for n, s in zip(names, scores) if n == cls]
        if len(cls_scores) == 0:
            class_counts.append(0.0)
            class_max_scores.append(0.0)
        else:
            class_counts.append(float(len(cls_scores)))
            class_max_scores.append(float(max(cls_scores)))

    feat = np.array(
        [num_det, overall_max, overall_mean, overall_std,
         best_nd_score, max_det_score, rank_in_model]
        + class_counts
        + class_max_scores,
        dtype=np.float32,
    )

    # ---------- 标签：best_model ----------
    bm_raw = str(row["best_model"]).strip()
    bm_key = bm_raw.lower()

    if bm_key not in MODEL_TO_ID:
        raise ValueError(
            f"Unknown best_model '{bm_raw}' for sample_token={row['sample_token']}"
        )
    label = MODEL_TO_ID[bm_key]

    sample_token = str(row["sample_token"])
    return feat, label, sample_token


def build_dataset_from_csv(csv_path: str):
    """
    从 CSV 构建 X, y, tokens.
    """
    df = pd.read_csv(csv_path)
    print(f"[Info] Loaded CSV with {len(df)} rows from {csv_path}")

    X_list = []
    y_list = []
    token_list = []

    for idx, row in df.iterrows():
        try:
            feat, lab, token = build_features_from_row(row)
            X_list.append(feat)
            y_list.append(lab)
            token_list.append(token)
        except Exception as e:
            print(f"[Warn] Skip row {idx} due to error: {e}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    tokens = np.array(token_list)

    print(f"[Info] Built dataset: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y, tokens


# ====================== MAIN VALIDATION ======================

def main():
    # 1. 构造数据
    X, y, tokens = build_dataset_from_csv(CSV_PATH)

    # 2. 加载模型
    print(f"[Info] Loading XGBoost model from: {MODEL_PATH}")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(MODEL_TO_ID),
    )
    model.load_model(MODEL_PATH)

    # 3. 预测
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    print(f"\n[Validation] Overall Accuracy: {acc:.4f}\n")

    print("[Classification Report]")
    print(
        classification_report(
            y,
            y_pred,
            target_names=[ID_TO_MODEL[i] for i in range(len(MODEL_TO_ID))],
        )
    )

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    print("[Confusion Matrix]")
    print(cm)

    # 4. 逐类别准确率
    print("\n[Per-Class Accuracy]")
    for cls_id in range(len(MODEL_TO_ID)):
        cls_name = ID_TO_MODEL[cls_id]
        mask = (y == cls_id)
        if mask.sum() == 0:
            print(f"  {cls_name:10s} : N/A (no samples)")
            continue
        cls_acc = (y_pred[mask] == y[mask]).mean()
        print(f"  {cls_name:10s} : {cls_acc:.4f} (N={mask.sum()})")

    # 5. 可选：导出 per-sample 结果，方便手动分析
    out_csv = "/home/frank/Pu/sci_ML/Tools_nuscenes/xgb_val_results_v2.csv"
    df_out = pd.DataFrame(
        {
            "sample_token": tokens,
            "y_true_id": y,
            "y_true_model": [ID_TO_MODEL[i] for i in y],
            "y_pred_id": y_pred,
            "y_pred_model": [ID_TO_MODEL[i] for i in y_pred],
        }
    )
    df_out.to_csv(out_csv, index=False)
    print(f"\n[Done] Per-sample prediction saved to: {out_csv}")


if __name__ == "__main__":
    main()
