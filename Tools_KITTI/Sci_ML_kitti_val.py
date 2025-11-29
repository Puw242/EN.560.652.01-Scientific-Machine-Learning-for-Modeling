#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test a trained 5-layer MLP on KITTI 'case' JSON files.

This script must be consistent with the training script:
- Same feature extraction
- Same 3-way label mapping
- Same dataset split logic (random_split + fixed seed)

Model path:
    /home/frank/Pu/sci_ML/Tools/case_mlp_3way.pth

Run:
    python /home/frank/Pu/sci_ML/kitti/test_case_mlp_3way.py
"""

# ====================== GLOBAL CONFIG ======================

CASE_DIR     = "/home/frank/Pu/sci_ML/kitti/Dataset_Prepare/selected_cases_60"
MODEL_PATH   = "/home/frank/Pu/sci_ML/Tools/case_mlp_3way_pibo_88_94.pth" 
# /home/frank/Pu/sci_ML/Tools/case_mlp_3way_pibo_88_94.pth
# /home/frank/Pu/sci_ML/Tools/case_mlp_3way_61_1.pth

RANDOM_SEED  = 42
BATCH_SIZE   = 16
VAL_RATIO    = 0.2

INPUT_DIM    = 15
HIDDEN_SIZES = [64, 64, 32]
NUM_CLASSES  = 3
NUM_HEADS    = 3

DEVICE       = "cuda"  # or "cpu"


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

    # labels
    y_car = encode_label_3way(int(tag_vec[0]))
    y_ped = encode_label_3way(int(tag_vec[1]))
    y_cyc = encode_label_3way(int(tag_vec[2]))
    labels = np.array([y_car, y_ped, y_cyc], dtype=np.int64)

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

    features = np.array(feat_car + feat_ped + feat_cyc, dtype=np.float32)
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
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        case = json.loads(s)
                        feat, lab = extract_features_and_labels_from_case(case)
                        self.samples.append((feat, lab))
                        break  # only first line

        print(f"[Dataset] Loaded {len(self.samples)} samples from {self.case_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat, lab = self.samples[idx]
        feat_t = torch.from_numpy(feat)   # [15]
        lab_t = torch.from_numpy(lab)     # [3]
        return feat_t, lab_t


def create_dataloaders(dataset: CaseDataset,
                       batch_size: int,
                       val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Split dataset into train/val using val_ratio, return two DataLoaders.
    Using random_split with a fixed seed to match training.
    """
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
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
        out_dim = num_heads * num_classes
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_flat = self.mlp(x)  # [B, num_heads * num_classes]
        logits = logits_flat.view(-1, self.num_heads, self.num_classes)
        return logits


# ====================== EVALUATION =========================

def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: torch.device,
             desc: str = "Eval") -> Tuple[float, float]:
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

    print(f"[{desc}] Loss: {avg_loss:.4f} | Overall Acc: {overall_acc:.4f}")
    return avg_loss, overall_acc


# ====================== MAIN ===============================

def main():
    set_seed(RANDOM_SEED)

    # 1. Load dataset
    dataset = CaseDataset(CASE_DIR)

    # 2. Create train/val loaders with same logic as training
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO
    )

    # 3. Build model and load weights
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = MLPModel(
        input_dim=INPUT_DIM,
        hidden_sizes=HIDDEN_SIZES,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[Info] Loaded model weights from: {MODEL_PATH}")

    # 4. Evaluate on train / val
    print("========= EVALUATION =========")
    evaluate(model, train_loader, device, desc="Train split")
    evaluate(model, val_loader, device, desc="Val split")

    # 5. (Optional) Evaluate on full dataset as 'test'
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    evaluate(model, full_loader, device, desc="Full dataset")


if __name__ == "__main__":
    main()
