#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage XGBoost training with PIBO-style weighting on nuScenes meta CSV.

Stage 0 (PIBO):
    - Uses physical information (best_nd_score, max_det_score, rank_in_model)
      to build sample_weight.
    - Trains XGBoost for a small number of boosting rounds with these weights.
    - We treat weighted mlogloss as "L_phys", and its CE proxy is mlogloss
      on the validation set, plus validation accuracy as proxy.

Stage 1 (Supervised Training):
    - Continues training from Stage 0 model (warm-start) with *uniform* weights.
    - Standard multi-class supervised learning.
    - Prints per-iteration train/val mlogloss + val accuracy.

Input CSV format (each row = 1 sample_token):

sample_token,num_detections,detection_names,detection_scores,boxes_lidar,
pred_labels,rank_in_model,best_model,best_nd_score,max_det_score

Example row (shortened):

05bc09f9...,10,
"['car', 'pedestrian', ...]",
"[0.8306, 0.8262, ...]",
"[[...], [...], ...]",
"[-1, -1, ...]",
16,BEVfusion,0.586518,0.8306
"""

import ast
import numpy as np
import pandas as pd

from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import xgboost as xgb


# ====================== CONFIG ======================

CSV_PATH = "/home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv"

RANDOM_SEED = 42
TEST_SIZE   = 0.2   # 20% as validation

# XGBoost rounds for two stages
STAGE0_ROUNDS = 10   # PIBO warm-up (you can tune)
STAGE1_ROUNDS = 40   # Supervised fine-tuning (you can tune)

# nuScenes main classes used in features
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

# best_model → integer label
MODEL_TO_ID = {
    "bevdal": 0,
    "bevfusion": 1,
    "transfusion": 2,   # we will map "TransFusion"/"Transfusion"/"transfusion" → lowercase
}

ID_TO_MODEL = {v: k for k, v in MODEL_TO_ID.items()}


# ====================== FEATURE & DATASET ======================

def build_features_from_row(row: pd.Series) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    从一行 CSV 构造:
        feat         : 特征向量 X
        label        : y = best_model 编码后的类别
        phys_vec[3]  : 物理信息向量 [best_nd_score, max_det_score, rank_in_model]

    特征结构:
      [ num_detections,
        overall_max_score, overall_mean_score, overall_std_score,
        best_nd_score, max_det_score, rank_in_model,
        counts for each class (10),
        max score for each class (10)
      ]
    """
    # ---------- detection_names / detection_scores ----------
    names = ast.literal_eval(row["detection_names"])
    scores = ast.literal_eval(row["detection_scores"])

    if len(names) != len(scores):
        raise ValueError(f"names 和 scores 长度不一致, sample_token={row['sample_token']}")

    scores_arr = np.asarray(scores, dtype=np.float32)

    num_det = float(row["num_detections"])
    if len(scores_arr) > 0:
        overall_max = float(scores_arr.max())
        overall_mean = float(scores_arr.mean())
        overall_std = float(scores_arr.std())
    else:
        overall_max = 0.0
        overall_mean = 0.0
        overall_std = 0.0

    best_nd_score = float(row.get("best_nd_score", 0.0))
    max_det_score = float(row.get("max_det_score", overall_max))
    rank_in_model = float(row.get("rank_in_model", 0.0))

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

    # ---------- phys-vector ----------
    phys_vec = np.array(
        [best_nd_score, max_det_score, rank_in_model],
        dtype=np.float32
    )

    return feat, label, phys_vec


def build_dataset_from_csv(csv_path: str):
    """
    从 CSV 构造:
      X: [N, D] 特征
      y: [N]    标签
      phys: [N, 3] 物理信息向量
    """
    df = pd.read_csv(csv_path)
    print(f"[Dataset] Loaded {len(df)} samples from {csv_path}")

    X_list = []
    y_list = []
    phys_list = []

    for idx, row in df.iterrows():
        try:
            feat, lab, phys = build_features_from_row(row)
            X_list.append(feat)
            y_list.append(lab)
            phys_list.append(phys)
        except Exception as e:
            print(f"[Warn] Skip row {idx} due to error: {e}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    phys = np.stack(phys_list, axis=0)

    print(f"[Info] Built dataset: X.shape = {X.shape}, y.shape = {y.shape}, phys.shape = {phys.shape}")
    return X, y, phys


# ====================== PIBO WEIGHT DESIGN ======================

# Part 1: 注意观察一下这里的 PIBO 结果的情况，这里是这样的结果。

def compute_pibo_sample_weight(phys_arr: np.ndarray) -> np.ndarray:
    """
    phys_arr: [N, 3] = [best_nd_score, max_det_score, rank_in_model]

    设计一个简单的 PIBO 权重：
      - best_nd_score 越高，样本越可靠
      - max_det_score 越高，说明模型对 top 检测更 confident
      - rank_in_model 越靠前（数字越小），说明是该模型更拿手的 case

    weight_i ~ 0.6 * best_nd + 0.2 * max_det + 0.2 * rank_score
    rank_score = 1 / (1 + rank)
    然后做一个截断，避免 0，范围大约在 [0.2, 2]。
    """
    best_nd = phys_arr[:, 0]
    max_det = phys_arr[:, 1]
    rank = phys_arr[:, 2]

    # 都假设在合理范围 (best_nd, max_det ∈ [0,1])
    rank_score = 1.0 / (1.0 + rank)  # rank 越大，score 越小

    raw = 0.6 * best_nd + 0.2 * max_det + 0.2 * rank_score
    weight = 0.2 + raw  # shift，避免完全 0

    weight = np.clip(weight, 0.2, 2.0)
    return weight.astype(np.float32)


# ====================== METRIC HELPERS ======================

def compute_val_acc_per_round(booster: xgb.Booster,
                              dval: xgb.DMatrix,
                              y_val: np.ndarray,
                              num_rounds: int,
                              offset: int = 0) -> List[float]:
    """
    计算每一轮的 validation accuracy.

    booster   : 训练好的 xgb.Booster
    dval      : validation DMatrix
    y_val     : validation labels
    num_rounds: 当前阶段的轮数
    offset    : booster 中前面已经存在的轮数（Stage 1 时需要加上 Stage 0 的轮数）

    对于第 i 轮 (0-based)，使用 iteration_range=(0, offset+i+1) 做预测。
    """
    acc_list = []
    for i in range(num_rounds):
        iter_end = offset + i + 1
        y_prob = booster.predict(dval, iteration_range=(0, iter_end))
        y_pred = y_prob.argmax(axis=1)
        acc = float((y_pred == y_val).mean())
        acc_list.append(acc)
    return acc_list


# ====================== MAIN TRAINING ======================

def main():
    np.random.seed(RANDOM_SEED)

    # 1. 构建数据
    X, y, phys = build_dataset_from_csv(CSV_PATH)

    # 2. train / val 划分（同时划分 phys）
    X_train, X_val, y_train, y_val, phys_train, phys_val = train_test_split(
        X,
        y,
        phys,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"[Split] Train: {len(X_train)}, Val: {len(X_val)}")

    # 3. 构造 DMatrix
    #  Stage 0：带 sample_weight 的 DMatrix
    w_train_stage0 = compute_pibo_sample_weight(phys_train)
    w_val_stage0   = compute_pibo_sample_weight(phys_val)

    dtrain_stage0 = xgb.DMatrix(X_train, label=y_train, weight=w_train_stage0)
    dval_stage0   = xgb.DMatrix(X_val,   label=y_val,   weight=w_val_stage0)

    #  Stage 1：不带权重 (uniform)
    dtrain_stage1 = xgb.DMatrix(X_train, label=y_train)
    dval_stage1   = xgb.DMatrix(X_val,   label=y_val)

    # 4. XGBoost 参数
    params = {
        "objective": "multi:softprob",
        "num_class": len(MODEL_TO_ID),
        "eval_metric": "mlogloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",  # 如果想用 GPU: "gpu_hist"
        "seed": RANDOM_SEED,
    }

    # ==================== Stage 0: PIBO ====================
    evals_result_stage0 = {}
    evals0 = [(dtrain_stage0, "train"), (dval_stage0, "val")]

    booster_stage0 = xgb.train(
        params=params,
        dtrain=dtrain_stage0,
        num_boost_round=STAGE0_ROUNDS,
        evals=evals0,
        evals_result=evals_result_stage0,
        verbose_eval=False,
    )

    # 计算每一轮的 Val Acc (proxy)
    val_acc_stage0 = compute_val_acc_per_round(
        booster_stage0,
        dval_stage0,
        y_val,
        num_rounds=STAGE0_ROUNDS,
        offset=0,
    )

    # -------- 打印 Stage 0 结果 --------
    print("\n========== Stage 0: PIBO (Physical-Information-Based Initialization) ==========")
    print("Epoch |  Train L_phys  |  Val CE (proxy)  |  Val Acc (proxy)")
    print("---------------------------------------------------------------")
    train_loss_list0 = evals_result_stage0["train"]["mlogloss"]
    val_loss_list0   = evals_result_stage0["val"]["mlogloss"]

    for epoch in range(1, STAGE0_ROUNDS + 1):
        tl = train_loss_list0[epoch - 1]
        vl = val_loss_list0[epoch - 1]
        acc = val_acc_stage0[epoch - 1]
        print(f"{epoch:5d} |    {tl:10.4f} |      {vl:10.4f} |    {acc:8.4f}")

    # ==================== Stage 1: Supervised Training ====================
    evals_result_stage1 = {}
    evals1 = [(dtrain_stage1, "train"), (dval_stage1, "val")]

    booster_stage1 = xgb.train(
        params=params,
        dtrain=dtrain_stage1,
        num_boost_round=STAGE1_ROUNDS,
        evals=evals1,
        evals_result=evals_result_stage1,
        verbose_eval=False,
        xgb_model=booster_stage0,  # 以 Stage 0 的模型为起点继续训练
    )

    # 计算 Stage 1 每一轮的 Val Acc
    val_acc_stage1 = compute_val_acc_per_round(
        booster_stage1,
        dval_stage1,
        y_val,
        num_rounds=STAGE1_ROUNDS,
        offset=STAGE0_ROUNDS,  # 注意偏移
    )

    # -------- 打印 Stage 1 结果 --------
    print("\n====================== Stage 1: Supervised Training ======================")
    print("Epoch |  Train CE (±Phys) |   Val CE       |  Val Acc")
    print("------------------------------------------------------")
    train_loss_list1 = evals_result_stage1["train"]["mlogloss"]
    val_loss_list1   = evals_result_stage1["val"]["mlogloss"]

    for epoch in range(1, STAGE1_ROUNDS + 1):
        tl = train_loss_list1[epoch - 1]
        vl = val_loss_list1[epoch - 1]
        acc = val_acc_stage1[epoch - 1]
        print(f"{epoch:5d} |      {tl:11.4f} |   {vl:10.4f} |  {acc:8.4f}")

    # ==================== Final Evaluation & Save ==========================
    # 用最终 booster 在 val 上做一次完整评估
    y_prob_val = booster_stage1.predict(dval_stage1)
    y_pred_val = y_prob_val.argmax(axis=1)
    acc_final = accuracy_score(y_val, y_pred_val)

    print("\n[Final Result] Val Accuracy after Stage 1: {:.4f}\n".format(acc_final))
    print("[Confusion Matrix]")
    print(confusion_matrix(y_val, y_pred_val))
    print("\n[Classification Report]")
    print(classification_report(
        y_val,
        y_pred_val,
        target_names=[ID_TO_MODEL[i] for i in range(len(MODEL_TO_ID))]
    ))

    # 保存最终模型
    model_save_path = "/home/frank/Pu/sci_ML/Tools_nuscenes/xgb_nuscenes_PIBO_two_stage.bin"
    booster_stage1.save_model(model_save_path)
    print(f"\n[Done] Two-stage XGBoost model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
