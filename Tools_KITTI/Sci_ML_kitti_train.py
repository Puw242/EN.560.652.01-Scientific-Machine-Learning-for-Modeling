#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a 5-layer MLP on 60 KITTI 'case' JSON files for 3-way classification
on (Car, Pedestrian, Cyclist) model choices.

Each case file looks like:

{
  "frame_id": "1014",
  "tag": "12",
  "tag_vector": [1, 2, 0],
  "Car": {...},
  "Pedestrian": {...},
  "Cyclist": {...}
}

We build features from per-class fields and predict 3 labels:
  y_car, y_ped, y_cyc \in {0,1,2}
where:
  0 -> None
  1 -> PartA2
  2 -> Other (PointRCNN / TED merged)

Loss = CE_car + CE_ped + CE_cyc.

Run:
    python /home/frank/Pu/sci_ML/kitti/train_case_mlp.py
"""

# ====================== GLOBAL CONFIG ======================

# Data directory: where your 60 case_XX_frame_XXXX_tagYY.json live
CASE_DIR = "/home/frank/Pu/sci_ML/kitti/Dataset_Prepare/selected_cases_60"

# Training hyperparameters
RANDOM_SEED   = 42
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
EPOCHS        = 30
VAL_RATIO     = 0.2  # 20% for validation

# MLP architecture
INPUT_DIM     = 15   # 3 classes * 5 features each
HIDDEN_SIZES  = [64, 64, 32]  # 5-layer: 3 hidden + 1 output + input layer
NUM_CLASSES   = 3    # 3-way classification per head
NUM_HEADS     = 3    # Car, Pedestrian, Cyclist

# Device
DEVICE        = "cuda"  # "cuda" or "cpu"


# ====================== IMPORTS ============================

import os
import json
import glob
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ====================== UTILITIES ==========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_label_3way(raw_value: int) -> int:
    """
    Map raw tag_vector element (0/1/2/3) to 3-way label:
      0 -> 0 (None)
      1 -> 1 (PartA2)
      2,3 -> 2 (Other: PointRCNN / TED)
    """
    if raw_value == 0:
        return 0
    elif raw_value == 1:
        return 1
    else:  # 2 or 3
        return 2


def encode_difficulty(diff_str):
    """
    Encode difficulty string "d0"/"d1"/"d2" to numeric value.
    If None or unknown, return 0.
    """
    if diff_str is None:
        return 0
    if diff_str == "d0":
        return 0
    if diff_str == "d1":
        return 1
    if diff_str == "d2":
        return 2
    # fallback
    return 0


def extract_features_and_labels_from_case(case_dict: dict):
    """
    Given a case dict, build 15-d feature vector and 3-d label vector.
    Features per class: [present, score, iou, det_idx, difficulty_code]
    Label: tag_vector mapped to 3-way classification using encode_label_3way.
    """
    tag_vec = case_dict.get("tag_vector", [0, 0, 0])
    if len(tag_vec) != 3:
        raise ValueError(f"tag_vector must be length 3, got {tag_vec}")

    # Build labels
    y_car = encode_label_3way(int(tag_vec[0]))
    y_ped = encode_label_3way(int(tag_vec[1]))
    y_cyc = encode_label_3way(int(tag_vec[2]))
    labels = np.array([y_car, y_ped, y_cyc], dtype=np.int64)

    # Helper: build features for one class
    def build_class_feature(cls_name: str):
        cls_info = case_dict.get(cls_name, {})
        present = float(bool(cls_info.get("present", False)))
        score = float(cls_info.get("score", 0.0) or 0.0)
        iou = float(cls_info.get("iou", 0.0) or 0.0)
        det_idx = float(cls_info.get("det_idx", 0) or 0)
        diff_code = float(encode_difficulty(cls_info.get("difficulty", None)))
        return [present, score, iou, det_idx, diff_code]

    feat_car = build_class_feature("Car")
    feat_ped = build_class_feature("Pedestrian")
    feat_cyc = build_class_feature("Cyclist")

    features = np.array(feat_car + feat_ped + feat_cyc, dtype=np.float32)  # 15-dim
    return features, labels


# ====================== DATASET ============================

class CaseDataset(Dataset):
    """
    Dataset that loads all case JSON/JSONL files from CASE_DIR.
    Each sample: (features[15], labels[3])
    """

    def __init__(self, case_dir: str):
        self.case_dir = case_dir
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self._load_all_cases()

    def _load_all_cases(self):
        # Support both .json and .jsonl
        json_files = sorted(glob.glob(os.path.join(self.case_dir, "*.json")))
        jsonl_files = sorted(glob.glob(os.path.join(self.case_dir, "*.jsonl")))
        all_files = json_files + jsonl_files

        if not all_files:
            raise FileNotFoundError(f"No .json or .jsonl files found in {self.case_dir}")

        for path in all_files:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    case = json.load(f)
                feat, lab = extract_features_and_labels_from_case(case)
                self.samples.append((feat, lab))
            else:
                # .jsonl 文件假设每个文件只有一行一个 case（如果有多行，可视情况修改）
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        case = json.loads(s)
                        feat, lab = extract_features_and_labels_from_case(case)
                        self.samples.append((feat, lab))
                        break  # 只用第一行

        print(f"[Dataset] Loaded {len(self.samples)} samples from {self.case_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat, lab = self.samples[idx]
        feat_t = torch.from_numpy(feat)         # float32 [15]
        lab_t = torch.from_numpy(lab)           # int64  [3]
        return feat_t, lab_t


def create_dataloaders(dataset: CaseDataset,
                       batch_size: int,
                       val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Split dataset into train/val using val_ratio, return two DataLoaders.
    """
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
    5-layer MLP: input -> hidden1 -> hidden2 -> hidden3 -> output
    Output dimension = NUM_HEADS * NUM_CLASSES, then reshaped to [B, 3, 3]
    """

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: List[int],
                 num_heads: int,
                 num_classes: int):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        # final layer
        out_dim = num_heads * num_classes
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        return: logits [B, num_heads, num_classes]
        """
        logits_flat = self.mlp(x)  # [B, num_heads * num_classes]
        logits = logits_flat.view(-1, self.num_heads, self.num_classes)
        return logits


# ====================== TRAIN / EVAL =======================

def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    ce_loss = nn.CrossEntropyLoss()

    for feats, labels in dataloader:
        feats = feats.to(device)                 # [B, 15]
        labels = labels.to(device)               # [B, 3] (Car, Ped, Cyc)

        optimizer.zero_grad()
        logits = model(feats)                    # [B, 3, 3]

        # Split heads: 0=Car, 1=Ped, 2=Cyc
        loss_car = ce_loss(logits[:, 0, :], labels[:, 0])
        loss_ped = ce_loss(logits[:, 1, :], labels[:, 1])
        loss_cyc = ce_loss(logits[:, 2, :], labels[:, 2])

        loss = loss_car + loss_ped + loss_cyc
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: str) -> Tuple[float, float]:
    """
    Return: (average_loss, overall_accuracy)
    overall_accuracy = mean of per-head accuracies.
    """
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0

    correct_car = 0
    correct_ped = 0
    correct_cyc = 0
    total_samples = 0

    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.to(device)
            labels = labels.to(device)

            logits = model(feats)  # [B, 3, 3]
            loss_car = ce_loss(logits[:, 0, :], labels[:, 0])
            loss_ped = ce_loss(logits[:, 1, :], labels[:, 1])
            loss_cyc = ce_loss(logits[:, 2, :], labels[:, 2])
            loss = loss_car + loss_ped + loss_cyc

            total_loss += loss.item()
            total_batches += 1

            preds = logits.argmax(dim=-1)  # [B, 3]
            correct_car += (preds[:, 0] == labels[:, 0]).sum().item()
            correct_ped += (preds[:, 1] == labels[:, 1]).sum().item()
            correct_cyc += (preds[:, 2] == labels[:, 2]).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(1, total_batches)
    if total_samples > 0:
        acc_car = correct_car / total_samples
        acc_ped = correct_ped / total_samples
        acc_cyc = correct_cyc / total_samples
        overall_acc = (acc_car + acc_ped + acc_cyc) / 3.0
    else:
        overall_acc = 0.0

    return avg_loss, overall_acc


# ====================== MAIN ===============================

def main():
    set_seed(RANDOM_SEED)

    # 1. Load dataset
    dataset = CaseDataset(CASE_DIR)

    # 2. Create train/val loaders
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO
    )

    # 3. Build model
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = MLPModel(
        input_dim=INPUT_DIM,
        hidden_sizes=HIDDEN_SIZES,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch:02d}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    # (可选) 保存模型
    save_path = os.path.join(CASE_DIR, "/home/frank/Pu/sci_ML/Tools/case_mlp_3way.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Done] Model saved to: {save_path}")


if __name__ == "__main__":
    main()
