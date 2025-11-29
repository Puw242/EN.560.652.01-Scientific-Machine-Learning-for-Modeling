#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a small MLP on nuScenes merged_top_samples CSV for 3-way model selection
(BEVDAL, BEVfusion, TransFusion) with a PIBO-like initialization.

Input CSV format (one row per sample_token):

sample_token,num_detections,detection_names,detection_scores,boxes_lidar,
pred_labels,rank_in_model,best_model,best_nd_score,max_det_score

Example row:
05bc09f9...,10,
"['car', 'pedestrian', ...]",
"[0.8306, 0.8262, ...]",
"[[x,y,z,l,w,h,yaw], ...]",
"[-1,...]", 16, BEVfusion, 0.586518, 0.8306

We:
  - Build a 20-d feature vector per row.
  - Label = argmax over {BEVDAL, BEVfusion, TransFusion}.
  - Stage 0 (PIBO): match prediction confidence with best_nd_score (physical signal).
  - Stage 1: supervised cross-entropy on best_model label.
"""

# ====================== GLOBAL CONFIG ======================

CSV_PATH       = "/home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv"

RANDOM_SEED    = 42
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-3
EPOCHS_STAGE0  = 5
EPOCHS_STAGE1  = 50
VAL_RATIO      = 0.2

USE_PHYS_IN_STAGE1 = False
LAMBDA_PHYS_STAGE1 = 0.1

INPUT_DIM      = 20   # we build 20-d features below
HIDDEN_SIZES   = [64, 64, 32]
NUM_CLASSES    = 3    # BEVDAL / BEVfusion / TransFusion

DEVICE         = "cuda"   # or "cpu"


# ====================== IMPORTS ============================

import ast
import csv
import math
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ====================== UTILITIES ==========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


MODEL_TO_ID = {
    "BEVDAL": 0,
    "BEVfusion": 1,
    "TransFusion": 2,
    "Transfusion": 2,
    "transfusion": 2,
}


def safe_literal_eval(s):
    """Parse a Python list stored as string."""
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return ast.literal_eval(s)


def build_features_from_row(row: dict) -> Tuple[np.ndarray, int]:
    """
    Given a CSV row, build a 20-dim feature vector and a single integer label.

    Feature order (20-dim):

    0: num_dets_norm (= num_detections / 10)
    1: max_score
    2: mean_score
    3: std_score
    4: min_score
    5: best_nd_score

    # per-group statistics (vehicles / VRU / static)
    6: vehicle_count_norm
    7: vehicle_mean_score
    8: vru_count_norm
    9: vru_mean_score
    10: static_count_norm
    11: static_mean_score

    # geometry statistics from boxes_lidar
    12: mean_x
    13: mean_y
    14: mean_z
    15: mean_l
    16: mean_w
    17: mean_h
    18: mean_dist (sqrt(x^2 + y^2))
    19: std_dist
    """

    # ---------- detection scores ----------
    num_det = int(row["num_detections"])
    scores = safe_literal_eval(row["detection_scores"])
    scores = [float(s) for s in scores] if scores else []
    if len(scores) == 0:
        max_score = 0.0
        mean_score = 0.0
        std_score = 0.0
        min_score = 0.0
    else:
        arr = np.array(scores, dtype=np.float32)
        max_score = float(arr.max())
        mean_score = float(arr.mean())
        std_score = float(arr.std())
        min_score = float(arr.min())

    num_dets_norm = num_det / 10.0  # since我们是top10, 这里大多是1.0

    best_nd_score = float(row.get("best_nd_score", 0.0) or 0.0)

    # ---------- per-group stats (by detection_names) ----------
    names = safe_literal_eval(row["detection_names"])
    # 对齐 scores 长度
    if len(names) > len(scores):
        names = names[:len(scores)]
    elif len(names) < len(scores):
        scores = scores[:len(names)]

    vehicle_classes = {"car", "truck", "bus", "trailer", "construction_vehicle"}
    vru_classes = {"pedestrian", "motorcycle", "bicycle"}
    static_classes = {"barrier", "traffic_cone"}

    veh_scores, vru_scores, static_scores = [], [], []

    for cls_name, sc in zip(names, scores):
        c = str(cls_name)
        if c in vehicle_classes:
            veh_scores.append(sc)
        elif c in vru_classes:
            vru_scores.append(sc)
        elif c in static_classes:
            static_scores.append(sc)

    def group_stats(grp_scores):
        if len(grp_scores) == 0:
            return 0.0, 0.0
        return len(grp_scores) / max(1.0, float(num_det)), float(np.mean(grp_scores))

    veh_cnt_norm, veh_mean_sc = group_stats(veh_scores)
    vru_cnt_norm, vru_mean_sc = group_stats(vru_scores)
    static_cnt_norm, static_mean_sc = group_stats(static_scores)

    # ---------- geometry statistics from boxes ----------
    boxes = safe_literal_eval(row["boxes_lidar"])
    xs, ys, zs, ls, ws, hs, dists = [], [], [], [], [], [], []

    for b in boxes:
        # each b: [x, y, z, l, w, h, yaw]
        if len(b) < 6:
            continue
        x, y, z = float(b[0]), float(b[1]), float(b[2])
        l, w, h = float(b[3]), float(b[4]), float(b[5])
        dist = math.sqrt(x * x + y * y)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ls.append(l)
        ws.append(w)
        hs.append(h)
        dists.append(dist)

    def safe_mean(arr_list):
        if len(arr_list) == 0:
            return 0.0
        return float(np.mean(arr_list))

    def safe_std(arr_list):
        if len(arr_list) == 0:
            return 0.0
        return float(np.std(arr_list))

    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    mean_z = safe_mean(zs)
    mean_l = safe_mean(ls)
    mean_w = safe_mean(ws)
    mean_h = safe_mean(hs)
    mean_dist = safe_mean(dists)
    std_dist = safe_std(dists)

    feat = np.array([
        num_dets_norm,
        max_score,
        mean_score,
        std_score,
        min_score,
        best_nd_score,
        veh_cnt_norm,
        veh_mean_sc,
        vru_cnt_norm,
        vru_mean_sc,
        static_cnt_norm,
        static_mean_sc,
        mean_x,
        mean_y,
        mean_z,
        mean_l,
        mean_w,
        mean_h,
        mean_dist,
        std_dist,
    ], dtype=np.float32)

    # ---------- label from best_model ----------
    bm = str(row["best_model"])
    if bm not in MODEL_TO_ID:
        raise ValueError(f"Unknown best_model '{bm}' in row with sample_token={row['sample_token']}")
    label = MODEL_TO_ID[bm]  # int in [0,2]

    return feat, label


# ====================== DATASET ============================

class NuScenesCSVCaseDataset(Dataset):
    """
    Dataset that loads merged_top_samples CSV.
    Each sample: (features[20], label:int)
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.samples: List[Tuple[np.ndarray, int]] = []
        self._load_csv()

    def _load_csv(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat, lab = build_features_from_row(row)
                self.samples.append((feat, lab))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat, lab = self.samples[idx]
        feat_t = torch.from_numpy(feat)          # [20] float32
        lab_t = torch.tensor(lab, dtype=torch.long)  # scalar label
        return feat_t, lab_t


def create_dataloaders(dataset: Dataset,
                       batch_size: int,
                       val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[Split] Train: {n_train}, Val: {n_val}")
    return train_loader, val_loader


# ====================== MODEL ==============================

class MLPModel(nn.Module):
    """
    Simple MLP: input -> hidden1 -> hidden2 -> hidden3 -> logits(3)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: List[int],
                 num_classes: int):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        return self.mlp(x)   # [B, num_classes]


# ====================== PHYSICAL LOSS ======================

def physical_loss(feats: torch.Tensor,
                  logits: torch.Tensor) -> torch.Tensor:
    """
    PIBO-style physical loss for nuScenes.

    Idea:
      - best_nd_score (feats[:, 5]) ∈ [0,1] 是 teacher ND score.
      - 模型输出 logits -> probs; p_max = max over 3 classes.
      - L_phys = MSE(p_max, best_nd_score).

    这样在 stage0 利用物理信息 (NDS) 让网络先学到
    “高质量样本 -> 高置信度” 的结构，而不用 label。
    """
    probs = F.softmax(logits, dim=-1)  # [B, 3]
    p_max, _ = probs.max(dim=-1)       # [B]
    nds = feats[:, 5].clamp(0.0, 1.0)  # best_nd_score

    return ((p_max - nds) ** 2).mean()


# ====================== TRAIN / EVAL =======================

def train_one_epoch_phys(model: nn.Module,
                         dataloader: DataLoader,
                         optimizer: torch.optim.Optimizer,
                         device: str) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for feats, labels in dataloader:
        feats = feats.to(device)

        optimizer.zero_grad()
        logits = model(feats)          # [B, 3]
        loss = physical_loss(feats, logits)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def train_one_epoch_sup(model: nn.Module,
                        dataloader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        device: str,
                        use_phys: bool = False,
                        lambda_phys: float = 0.1) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    ce_loss = nn.CrossEntropyLoss()

    for feats, labels in dataloader:
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(feats)          # [B, 3]
        loss_sup = ce_loss(logits, labels)

        if use_phys:
            loss_phys = physical_loss(feats, logits)
            loss = loss_sup + lambda_phys * loss_phys
        else:
            loss = loss_sup

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: str) -> Tuple[float, float]:
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.to(device)
            labels = labels.to(device)

            logits = model(feats)          # [B, 3]
            loss = ce_loss(logits, labels)
            total_loss += loss.item()
            total_batches += 1

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(1, total_batches)
    acc = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, acc


# ====================== MAIN ===============================

def main():
    set_seed(RANDOM_SEED)

    # 1. Load dataset
    dataset = NuScenesCSVCaseDataset(CSV_PATH)

    # 2. Dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO
    )

    # 3. Model
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = MLPModel(
        input_dim=INPUT_DIM,
        hidden_sizes=HIDDEN_SIZES,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------- Stage 0 ----------
    print("\n========== Stage 0: PIBO (Physical-Information-Based Initialization) ==========")
    print("Epoch |  Train L_phys  |  Val CE (proxy)  |  Val Acc (proxy)")
    print("---------------------------------------------------------------")
    for epoch in range(1, EPOCHS_STAGE0 + 1):
        train_loss_phys = train_one_epoch_phys(model, train_loader, optimizer, device)
        val_loss_proxy, val_acc_proxy = evaluate(model, val_loader, device)
        print(f"{epoch:5d} |    {train_loss_phys:10.4f} |      {val_loss_proxy:10.4f} |    {val_acc_proxy:8.4f}")

    # ---------- Stage 1 ----------
    print("\n====================== Stage 1: Supervised Training ======================")
    print("Epoch |  Train CE (±Phys) |   Val CE       |  Val Acc")
    print("------------------------------------------------------")
    for epoch in range(1, EPOCHS_STAGE1 + 1):
        train_loss = train_one_epoch_sup(
            model,
            train_loader,
            optimizer,
            device,
            use_phys=USE_PHYS_IN_STAGE1,
            lambda_phys=LAMBDA_PHYS_STAGE1
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"{epoch:5d} |      {train_loss:11.4f} |   {val_loss:10.4f} |  {val_acc:8.4f}")

    save_path = "/home/frank/Pu/sci_ML/Tools/nusc_case_mlp_pibo.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[Done] nuScenes PIBO Model saved to: {save_path}")


if __name__ == "__main__":
    main()
