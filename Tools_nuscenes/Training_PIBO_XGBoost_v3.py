#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost + PIBO two-stage training on nuScenes meta CSV.

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

Pipeline:
  Stage 0 (PIBO): use physical information to build sample_weight_phys
                  and train a first booster.
  Stage 1: continue training from Stage 0 booster with (optionally) weaker
           or uniform weights, fully supervised.

Label: best_model ∈ {BEVDAL, BEVfusion, TransFusion}.
"""

import ast
import numpy as np
import pandas as pd

from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb


# ====================== CONFIG ======================

CSV_PATH = "/home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv"

RANDOM_SEED = 42
TEST_SIZE   = 0.2   # 20% for validation

# XGBoost 参数
STAGE0_ROUNDS = 80   # PIBO 阶段的树数量
STAGE1_ROUNDS = 120  # 在 Stage0 基础上继续训练的树数量

# nuScenes 10 个类别
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

# best_model → label
MODEL_TO_ID = {
    "bevdal": 0,
    "bevfusion": 1,
    "transfusion": 2,
    "transfusion": 2,    # 保守写两遍也无所谓
    "transfusion ": 2,   # 防止奇怪空格
    "TransFusion": 2,
    "Transfusion": 2,
}
# 为了更稳妥，统一 lower() 时再查表
ID_TO_MODEL = {v: k for k, v in MODEL_TO_ID.items()}


# ====================== FEATURE + PIBO WEIGHT ======================

def build_features_and_phys_weight_from_row(row: pd.Series) -> Tuple[np.ndarray, int, float]:
    """
    从一行 CSV 构造:
      - 特征向量 feat ∈ R^27
      - 标签 label ∈ {0,1,2}
      - 物理权重 w_phys ≥ 0 (用于 PIBO, sample_weight)

    特征结构:
      [ num_detections,
        overall_max_score, overall_mean_score, overall_std_score,
        best_nd_score, max_det_score, rank_in_model,
        counts for each class (10),
        max score for each class (10)
      ]

    物理权重构造 (PIBO 思路):
      w_phys = α * best_nd_score + β * max_det_score + γ * (1 / (1 + rank_in_model))
      然后 clip 到 [w_min, w_max]，最后归一化。
    """
    # ---------- parse detection_names / detection_scores ----------
    names = ast.literal_eval(row["detection_names"])
    scores = ast.literal_eval(row["detection_scores"])
    assert len(names) == len(scores), "names 和 scores 长度不一致"

    scores_arr = np.asarray(scores, dtype=np.float32)

    # basic statistics
    num_det = float(row["num_detections"])
    overall_max = float(scores_arr.max()) if len(scores_arr) > 0 else 0.0
    overall_mean = float(scores_arr.mean()) if len(scores_arr) > 0 else 0.0
    overall_std = float(scores_arr.std()) if len(scores_arr) > 0 else 0.0

    # physics-related meta
    best_nd_score = float(row.get("best_nd_score", 0.0) or 0.0)
    max_det_score = float(row.get("max_det_score", overall_max) or 0.0)
    rank_in_model = float(row.get("rank_in_model", 0.0) or 0.0)

    # ---------- per-class count & max score ----------
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

    # ---------- label ----------
    bm_raw = str(row["best_model"]).strip()
    bm_key = bm_raw.lower()
    if bm_key not in MODEL_TO_ID:
        raise ValueError(
            f"Unknown best_model '{bm_raw}' for sample_token={row['sample_token']}"
        )
    label = MODEL_TO_ID[bm_key]

    # ---------- PIBO sample_weight (物理权重) ----------
    # 物理直觉：
    #  - best_nd_score 越大 (0~1)，说明该模型在该 sample 上整体 quality 更高 → 权重↑
    #  - max_det_score 越大，说明 top detection 更 confident → 权重↑
    #  - rank_in_model 越小 (top rank)，说明该 sample 是该模型的“代表性样本” → 1/(1+rank) 越大 → 权重↑
    #
    # w_phys_raw = α*best_nd_score + β*max_det_score + γ*(1/(1+rank))
    # 这里 α, β, γ 可以微调；先给一个合理且简单的版本。
    alpha = 0.5
    beta  = 0.3
    gamma = 0.2

    inv_rank = 1.0 / (1.0 + rank_in_model)  # rank 从 1,2,... 开始时, rank=1 -> 0.5, rank=10 -> ~0.09

    w_phys_raw = alpha * best_nd_score + beta * max_det_score + gamma * inv_rank

    # 简单 clip，防止极端值
    w_min, w_max = 0.05, 2.0
    w_phys = float(np.clip(w_phys_raw, w_min, w_max))

    return feat, label, w_phys


def build_dataset_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 CSV 构建:
      X: [N, D] 特征
      y: [N] 标签
      w_phys: [N] 物理权重 (用于 PIBO)
    """
    df = pd.read_csv(csv_path)
    print(f"[Info] Loaded CSV with {len(df)} rows from {csv_path}")

    X_list = []
    y_list = []
    w_list = []

    for idx, row in df.iterrows():
        try:
            feat, lab, w_phys = build_features_and_phys_weight_from_row(row)
            X_list.append(feat)
            y_list.append(lab)
            w_list.append(w_phys)
        except Exception as e:
            print(f"[Warn] Skip row {idx} due to error: {e}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)

    # 为了让权重整体 scale 稍微好一点（不会太大/太小），做一个简单归一化
    mean_w = float(w.mean())
    if mean_w > 0:
        w = w / mean_w

    print(f"[Info] Built dataset: X.shape = {X.shape}, y.shape = {y.shape}, w_phys.mean = {w.mean():.4f}")
    return X, y, w


# ====================== MAIN: XGBoost + PIBO ======================

def main():
    np.random.seed(RANDOM_SEED)

    # 1. 构建数据
    X, y, w_phys = build_dataset_from_csv(CSV_PATH)

    # 2. train / val 划分 (注意要同时划分权重)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X,
        y,
        w_phys,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"[Split] Train: {len(X_train)}, Val: {len(X_val)}")

    # 3. 构造 DMatrix
    dtrain_stage0 = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval_stage0   = xgb.DMatrix(X_val,   label=y_val)

    # Stage 1 可以选：
    #  - 用统一权重 1.0 (完全监督)，或者
    #  - 用较弱的物理权重 (例如 sqrt(w_phys))，这里采用统一权重 1.0 简化。
    dtrain_stage1 = xgb.DMatrix(X_train, label=y_train)
    dval_stage1   = xgb.DMatrix(X_val,   label=y_val)

    # 4. XGBoost 参数
    params = {
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": len(ID_TO_MODEL),
        "eval_metric": "mlogloss",
        # 如果要用 GPU，可以改成:
        # "tree_method": "gpu_hist",
        "tree_method": "hist",
        "seed": RANDOM_SEED,
    }

    # ================== Stage 0: PIBO (物理权重训练) ==================
    print("\n========== Stage 0: PIBO (Physical-Information-Based Initialization) ==========")
    evals_result_0 = {}
    watchlist0 = [(dtrain_stage0, "train"), (dval_stage0, "val")]

    booster_stage0 = xgb.train(
        params=params,
        dtrain=dtrain_stage0,
        num_boost_round=STAGE0_ROUNDS,
        evals=watchlist0,
        evals_result=evals_result_0,
        verbose_eval=10,  # 每 10 棵树打印一次
    )

    # Stage 0 验证集表现
    y_val_prob0 = booster_stage0.predict(dval_stage0)
    y_val_pred0 = np.argmax(y_val_prob0, axis=1)
    acc0 = accuracy_score(y_val, y_val_pred0)
    print(f"\n[Stage 0 Result] Val Accuracy: {acc0:.4f}")
    print("[Stage 0 Confusion Matrix]")
    print(confusion_matrix(y_val, y_val_pred0))

    # ================== Stage 1: Supervised Fine-tuning ==================
    print("\n====================== Stage 1: Supervised Fine-tuning ======================")
    evals_result_1 = {}
    watchlist1 = [(dtrain_stage1, "train"), (dval_stage1, "val")]

    booster_stage1 = xgb.train(
        params=params,
        dtrain=dtrain_stage1,
        num_boost_round=STAGE1_ROUNDS,
        evals=watchlist1,
        evals_result=evals_result_1,
        verbose_eval=10,
        xgb_model=booster_stage0,  # 关键：在 Stage 0 booster 上继续训练
    )

    # Stage 1 验证集表现
    y_val_prob1 = booster_stage1.predict(dval_stage1)
    y_val_pred1 = np.argmax(y_val_prob1, axis=1)
    acc1 = accuracy_score(y_val, y_val_pred1)
    print(f"\n[Stage 1 Result] Val Accuracy: {acc1:.4f}\n")

    print("[Stage 1 Classification Report]")
    print(classification_report(
        y_val,
        y_val_pred1,
        target_names=[ID_TO_MODEL[i] for i in range(len(ID_TO_MODEL))]
    ))

    print("[Stage 1 Confusion Matrix]")
    print(confusion_matrix(y_val, y_val_pred1))

    # 额外打印一些简要的 PIBO → Stage1 收敛对比
    print("\n[Training Log Summary]")
    print(f"  Stage 0 train mlogloss last: {evals_result_0['train']['mlogloss'][-1]:.4f}")
    print(f"  Stage 0 val   mlogloss last: {evals_result_0['val']['mlogloss'][-1]:.4f}")
    print(f"  Stage 1 train mlogloss last: {evals_result_1['train']['mlogloss'][-1]:.4f}")
    print(f"  Stage 1 val   mlogloss last: {evals_result_1['val']['mlogloss'][-1]:.4f}")

    # ================== Save final booster ==================
    save_path = "/home/frank/Pu/sci_ML/Tools_nuscenes/xgb_nuscenes_PIBO_booster.bin"
    booster_stage1.save_model(save_path)
    print(f"\n[Done] PIBO-XGBoost model (Stage1 booster) saved to: {save_path}")


if __name__ == "__main__":
    main()
