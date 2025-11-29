#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train an XGBoost classifier on nuScenes meta CSV.

Input CSV format (each row = 1 sample_token):

sample_token,num_detections,detection_names,detection_scores,boxes_lidar,
pred_labels,rank_in_model,best_model,best_nd_score,max_det_score

Example (shortened):
05bc09f9...,10,
"['car', 'pedestrian', ...]",
"[0.8306, 0.8262, ...]",
"[[...], [...], ...]",
"[-1, -1, ...]",
16,BEVfusion,0.586518,0.8306

We build features from detection_names / detection_scores and metadata, then
predict best_model ∈ {BEVDAL, BEVfusion, TransFusion}.
"""

import ast
import numpy as np
import pandas as pd

from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb


# ====================== CONFIG ======================

# TODO: 根据你的实际文件名改这里
CSV_PATH = "/home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv"

RANDOM_SEED = 42
TEST_SIZE   = 0.2   # 20% 作为验证集

# nuScenes 中的 10 个主要类别（可以按需要修改/扩展）
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

# best_model 映射为整数标签
MODEL_TO_ID = {
    "bevdal": 0,
    "bevfusion": 1,
    # "transfusion": 2,  # 兼容 "Transfusion" / "TransFusion"
    "TransFusion": 2,
    "Transfusion": 2,
    "transfusion": 2,
}

ID_TO_MODEL = {v: k for k, v in MODEL_TO_ID.items()}


# ====================== FEATURE BUILDING ======================

def build_features_from_row(row: pd.Series) -> Tuple[np.ndarray, int]:
    """
    从一行 CSV 构造特征向量 X 和标签 y。

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

    # 防止乱序：确保长度一致
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

    return feat, label


def build_dataset_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 CSV 构建 X, y.
    """
    df = pd.read_csv(csv_path)
    print(f"[Info] Loaded CSV with {len(df)} rows from {csv_path}")

    X_list = []
    y_list = []

    for idx, row in df.iterrows():
        try:
            feat, lab = build_features_from_row(row)
            X_list.append(feat)
            y_list.append(lab)
        except Exception as e:
            print(f"[Warn] Skip row {idx} due to error: {e}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print(f"[Info] Built dataset: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y


# ====================== MAIN TRAINING ======================

def main():
    np.random.seed(RANDOM_SEED)

    # 1. 构建数据
    X, y = build_dataset_from_csv(CSV_PATH)

    # 2. train / val 划分
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"[Split] Train: {len(X_train)}, Val: {len(X_val)}")

    # 3. 定义 XGBoost 模型
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(MODEL_TO_ID),
        eval_metric="mlogloss",
        tree_method="hist",      # 如果要用 GPU，可以改成 "gpu_hist"
        random_state=RANDOM_SEED,
    )

    # 4. 训练（带 eval_set，可以看收敛情况）
    # model.fit(
    #     X_train,
    #     y_train,
    #     eval_set=[("train", X_train, y_train), ("val", X_val, y_val)],
    #     verbose=False,  # 想看每轮日志可以改成 True
    # )
    
    model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    # 如果你在 model 里没写 eval_metric，可以这里写：
    # eval_metric="mlogloss",
    verbose=False,  # 想看每一轮的 log 就改 True
    )

    # 5. 验证集评估
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n[Result] Val Accuracy: {acc:.4f}\n")

    print("[Classification Report]")
    print(classification_report(
        y_val,
        y_pred,
        target_names=[ID_TO_MODEL[i] for i in range(len(MODEL_TO_ID))]
    ))

    print("[Confusion Matrix]")
    print(confusion_matrix(y_val, y_pred))

    # 6. 特征重要性
    importance = model.feature_importances_
    feature_names = (
        [
            "num_det",
            "overall_max",
            "overall_mean",
            "overall_std",
            "best_nd_score",
            "max_det_score",
            "rank_in_model",
        ]
        + [f"count_{c}" for c in NUSC_CLASSES]
        + [f"maxscore_{c}" for c in NUSC_CLASSES]
    )

    # 排个序看一下前 20 个最重要特征
    idx_sorted = np.argsort(-importance)
    print("\n[Top 20 Feature Importances]")
    for i in idx_sorted[:20]:
        print(f"{feature_names[i]:20s} : {importance[i]:.4f}")

    # 7. 可选：保存模型
    model_save_path = "/home/frank/Pu/sci_ML/Tools_nuscenes/xgb_nuscenes_bestmodel_XGBoost.bin"
    model.save_model(model_save_path)
    print(f"\n[Done] XGBoost model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
