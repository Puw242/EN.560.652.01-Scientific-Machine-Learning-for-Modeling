#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost + PIBO (Physical-Information-Based Optimization) on nuScenes meta CSV.

PIBO 思路（针对 XGBoost）：
  - 利用物理相关指标：
        best_nd_score (检测整体质量)
        overall_max   (该 sample 中最高的 detection score)
        max_det_score (预先提取的最大检测得分，一般与 overall_max 一致/相近)
    来构造一个 “物理置信度” s_i，然后转成 sample_weight w_i。
  - s_i 越大，说明该场景在物理/感知层面越可靠，对“最佳模型”的选择越有参考价值，
    在训练中给更高权重（w_i 更大），实现基于物理信息的优化（PIBO）。

特征结构 (每一行 CSV → 一个样本):
  X: [ num_detections,
       overall_max_score, overall_mean_score, overall_std_score,
       best_nd_score, max_det_score, rank_in_model,
       counts for each class (10),
       max score for each class (10)
     ]
  y: best_model ∈ {BEVDAL, BEVfusion, TransFusion}

训练输出：
  ====================== XGBoost + PIBO Training ======================
  Iter |  Train mlogloss  |  Val mlogloss  |  Val Acc
  --------------------------------------------------------------
      1 |           ...   |         ...    |   ...
      2 |           ...   |         ...    |   ...
"""

import ast
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb


# ====================== CONFIG ======================

CSV_PATH = "/home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv"

RANDOM_SEED = 42
TEST_SIZE   = 0.2   # 20% 作为验证集
NUM_BOOST_ROUND = 50  # 提升轮数（可以调）

# nuScenes 中的 10 个主要类别（顺序要和之前保持一致）
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
    "transfusion": 2,   # 兼容 TransFusion / Transfusion / transfusion
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


def build_dataset_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从 CSV 构建 X, y 以及 feature_names.
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

    # 构造 feature_names（顺序要和 feat 中一致）
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

    return X, y, feature_names


# ====================== PIBO SAMPLE WEIGHT ======================

def minmax_norm(x: np.ndarray) -> np.ndarray:
    """简单的 min-max 归一化到 [0,1]。"""
    x_min = x.min()
    x_max = x.max()
    denom = max(x_max - x_min, 1e-6)
    return (x - x_min) / denom


def compute_pibo_weights(
    X: np.ndarray,
    feature_names: List[str],
    alpha_best_nd: float = 0.5,
    alpha_overall_max: float = 0.3,
    alpha_max_det: float = 0.2,
) -> np.ndarray:
    """
    基于物理信息构造 PIBO sample_weight.

    思路：
      s_i = 0.5 * norm(best_nd_score)
          + 0.3 * norm(overall_max)
          + 0.2 * norm(max_det_score)

      然后：
      w_i = 0.5 + s_i         （范围大致在 [0.5, 1.5]）
    """
    idx_best_nd = feature_names.index("best_nd_score")
    idx_overall_max = feature_names.index("overall_max")
    idx_max_det = feature_names.index("max_det_score")

    best_nd = X[:, idx_best_nd]
    overall_max = X[:, idx_overall_max]
    max_det = X[:, idx_max_det]

    best_nd_n = minmax_norm(best_nd)
    overall_max_n = minmax_norm(overall_max)
    max_det_n = minmax_norm(max_det)

    s = (
        alpha_best_nd * best_nd_n
        + alpha_overall_max * overall_max_n
        + alpha_max_det * max_det_n
    )

    weights = 0.5 + s  # PIBO: 物理置信度越高，权重越大
    return weights.astype(np.float32)


# ====================== METRICS ======================

def multi_logloss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    """
    手动计算多分类 logloss:
      -1/N ∑ log p(y_i | x_i)
    """
    n = y_true.shape[0]
    p = proba[np.arange(n), y_true]
    p = np.clip(p, eps, 1.0)
    return float(-np.mean(np.log(p)))


# ====================== MAIN (XGBoost + PIBO) ======================

def main():
    np.random.seed(RANDOM_SEED)

    # 1. 构建数据
    X, y, feature_names = build_dataset_from_csv(CSV_PATH)

    # 2. 计算基于物理信息的 PIBO 权重
    all_weights = compute_pibo_weights(X, feature_names)
    # 可以打印一下范围
    print(
        f"[PIBO] sample_weight range: min={all_weights.min():.4f}, max={all_weights.max():.4f}"
    )

    # 3. train / val 划分（确保权重一起划分）
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X,
        y,
        all_weights,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"[Split] Train: {len(X_train)}, Val: {len(X_val)}")

    # 4. 构造 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    # 5. XGBoost 参数
    params = {
        "objective": "multi:softprob",
        "num_class": len(MODEL_TO_ID),
        "eval_metric": "mlogloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",   # 有 GPU 可改为 "gpu_hist"
        "seed": RANDOM_SEED,
    }

    # 6. 逐轮训练 + 手动计算 train / val mlogloss + val acc
    print("\n====================== XGBoost + PIBO Training ======================")
    print("Iter |  Train mlogloss  |  Val mlogloss  |  Val Acc")
    print("--------------------------------------------------------------")

    booster = None

    for it in range(1, NUM_BOOST_ROUND + 1):
        # 每轮只训练 1 棵树，累计训练 NUM_BOOST_ROUND 轮
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=1,
            xgb_model=booster,
            verbose_eval=False,
        )

        # 预测概率
        train_proba = booster.predict(dtrain)
        val_proba = booster.predict(dval)

        # 计算 mlogloss / acc
        train_loss = multi_logloss(y_train, train_proba)
        val_loss = multi_logloss(y_val, val_proba)
        val_pred = np.argmax(val_proba, axis=1)
        val_acc = accuracy_score(y_val, val_pred)

        print(
            f"{it:4d} |        {train_loss:10.4f} |      {val_loss:10.4f} |   {val_acc:7.4f}"
        )

    # 7. 训练结束后，做一次完整的验证评估
    final_proba = booster.predict(dval)
    final_pred = np.argmax(final_proba, axis=1)
    final_acc = accuracy_score(y_val, final_pred)

    print(f"\n[Final Result] Val Accuracy: {final_acc:.4f}\n")

    print("[Classification Report]")
    print(
        classification_report(
            y_val,
            final_pred,
            target_names=[ID_TO_MODEL[i] for i in range(len(MODEL_TO_ID))],
        )
    )

    print("[Confusion Matrix]")
    print(confusion_matrix(y_val, final_pred))

    # 8. 特征重要性（基于 gain）
    importance = booster.get_score(importance_type="gain")
    # importance 是 dict: {feature_name: importance_value}
    print("\n[Top Feature Importances] (gain)")
    sorted_items = sorted(importance.items(), key=lambda x: -x[1])
    for name, val in sorted_items[:20]:
        print(f"{name:20s} : {val:.4f}")

    # 9. 保存模型
    model_save_path = "/home/frank/Pu/sci_ML/Tools_nuscenes/xgb_nuscenes_bestmodel_XGBoost_PIBO.bin"
    booster.save_model(model_save_path)
    print(f"\n[Done] XGBoost + PIBO model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
