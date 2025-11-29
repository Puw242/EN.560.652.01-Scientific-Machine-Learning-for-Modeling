#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Informed Router Training on 60 KITTI Cases

- Data:   /home/frank/Pu/sci_ML/kitti/Dataset_Prepare/selected_cases_60/*.json
- Model:  5-layer MLP (shared encoder) + 3 classification heads + 1 physics head
- Loss:   L = ALPHA * L_match + BETA * L_phys
          where L_match = CE_car + CE_ped + CE_cyc
                L_phys  = SmoothL1(rho_hat, rho_star) + SmoothL1(kappa_hat, kappa_star)

Tag format in each JSON:
  "tag_vector": [car_label, ped_label, cyc_label]
  labels are integers in {0,1,2,3}:
    0: None
    1: PartA2
    2: PointRCNN
    3: TED
"""

import os
import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ================== 全局配置 / Hyper-parameters ==================

DATA_DIR = "/home/frank/Pu/sci_ML/kitti/Dataset_Prepare/selected_cases_60"
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "case_mlp_3way_pinn.pth")  # 新模型文件名

BATCH_SIZE = 16
NUM_EPOCHS = 10          # 按你最新要求：训练 10 轮
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

NUM_CLASSES = 4          # tag_vector 里的 0/1/2/3
EMBED_DIM = 64           # MLP 中间维度
ALPHA_MATCH = 1.0        # L_match 权重
BETA_PHYS = 0.1          # L_phys 权重（可调）

TRAIN_RATIO = 0.8        # 48 train / 12 val

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ================== Dataset 定义 ==================


class KittiCaseDataset(Dataset):
    """
    从 selected_cases_60 目录读取所有 *.json，
    每个 case 生成一条样本：
      - x:  9 维特征 [Car_score, Car_iou, Car_present,
                       Ped_score, Ped_iou, Ped_present,
                       Cyc_score, Cyc_iou, Cyc_present]
      - y:  3 维 label 向量 [y_car, y_ped, y_cyc] (每个都是 0~3 的 int)
      - physics targets:
          rho_star   ∈ [0,1]  ~  present 的比例  (#present / 3)
          kappa_star ∈ [0,1]  ~  present 类的平均 IoU (无 present 时设为 0)
    """

    def __init__(self, data_dir: str):
        super().__init__()
        self.samples = []

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)

            tag_vec = obj.get("tag_vector", [0, 0, 0])
            if len(tag_vec) != 3:
                raise ValueError(f"tag_vector length != 3 in file: {fpath}")

            # 读出 Car/Ped/Cyc 三个子字典
            car = obj.get("Car", {})
            ped = obj.get("Pedestrian", {})
            cyc = obj.get("Cyclist", {})

            # helper: safe get
            def _get_score_iou_present(d):
                present = bool(d.get("present", False))
                score = d.get("score", 0.0)
                iou = d.get("iou", 0.0)
                score = 0.0 if score is None else float(score)
                iou = 0.0 if iou is None else float(iou)
                return score, iou, float(present)

            car_score, car_iou, car_present = _get_score_iou_present(car)
            ped_score, ped_iou, ped_present = _get_score_iou_present(ped)
            cyc_score, cyc_iou, cyc_present = _get_score_iou_present(cyc)

            # 9 维 feature 向量
            feat = [
                car_score, car_iou, car_present,
                ped_score, ped_iou, ped_present,
                cyc_score, cyc_iou, cyc_present,
            ]

            # 三个分类 label
            y_car, y_ped, y_cyc = tag_vec

            # ================== Physics targets (rho*, kappa*) ==================
            # rho*: actor density proxy ~ present 的数量 / 3
            present_flags = [car_present, ped_present, cyc_present]
            rho_star = sum(present_flags) / 3.0

            # kappa*: clarity proxy ~ present 类的平均 IoU
            iou_list = []
            if car_present > 0.5:
                iou_list.append(car_iou)
            if ped_present > 0.5:
                iou_list.append(ped_iou)
            if cyc_present > 0.5:
                iou_list.append(cyc_iou)
            if len(iou_list) > 0:
                kappa_star = sum(iou_list) / len(iou_list)
            else:
                kappa_star = 0.0

            self.samples.append({
                "frame_id": obj.get("frame_id", ""),
                "feat": feat,
                "y_car": int(y_car),
                "y_ped": int(y_ped),
                "y_cyc": int(y_cyc),
                "rho_star": float(rho_star),
                "kappa_star": float(kappa_star),
            })

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.tensor(s["feat"], dtype=torch.float32)
        y_car = torch.tensor(s["y_car"], dtype=torch.long)
        y_ped = torch.tensor(s["y_ped"], dtype=torch.long)
        y_cyc = torch.tensor(s["y_cyc"], dtype=torch.long)

        rho_star = torch.tensor(s["rho_star"], dtype=torch.float32)
        kappa_star = torch.tensor(s["kappa_star"], dtype=torch.float32)

        return x, y_car, y_ped, y_cyc, rho_star, kappa_star


# ================== 模型定义：5 层 MLP + 3 分类头 + 物理头 ==================


class PINNRouterMLP(nn.Module):
    """
    Shared 5-layer MLP encoder + 3 classification heads + 1 physics head.
    """

    def __init__(self, input_dim: int = 9, embed_dim: int = EMBED_DIM,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        # 5-layer MLP encoder (shared)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        # 3 classification heads
        self.head_car = nn.Linear(embed_dim, num_classes)
        self.head_ped = nn.Linear(embed_dim, num_classes)
        self.head_cyc = nn.Linear(embed_dim, num_classes)

        # physics head: predicts [rho_hat, kappa_hat]
        self.head_phys = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 9)
        returns:
          logits_car, logits_ped, logits_cyc, phys_pred (B,2)
        """
        h = self.encoder(x)
        logits_car = self.head_car(h)
        logits_ped = self.head_ped(h)
        logits_cyc = self.head_cyc(h)
        phys_pred = self.head_phys(h)  # [rho_hat, kappa_hat]
        return logits_car, logits_ped, logits_cyc, phys_pred


# ================== 训练 / 验证函数 ==================


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_ce: nn.Module,
    criterion_smooth: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y_car, y_ped, y_cyc, rho_star, kappa_star in dataloader:
        x = x.to(DEVICE)
        y_car = y_car.to(DEVICE)
        y_ped = y_ped.to(DEVICE)
        y_cyc = y_cyc.to(DEVICE)
        rho_star = rho_star.to(DEVICE)
        kappa_star = kappa_star.to(DEVICE)

        optimizer.zero_grad()
        logits_car, logits_ped, logits_cyc, phys_pred = model(x)

        # classification loss (L_match)
        loss_car = criterion_ce(logits_car, y_car)
        loss_ped = criterion_ce(logits_ped, y_ped)
        loss_cyc = criterion_ce(logits_cyc, y_cyc)
        loss_match = loss_car + loss_ped + loss_cyc

        # physics loss (L_phys)
        rho_hat = phys_pred[:, 0]
        kappa_hat = phys_pred[:, 1]
        loss_rho = criterion_smooth(rho_hat, rho_star)
        loss_kappa = criterion_smooth(kappa_hat, kappa_star)
        loss_phys = loss_rho + loss_kappa

        loss = ALPHA_MATCH * loss_match + BETA_PHYS * loss_phys
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_ce: nn.Module,
    criterion_smooth: nn.Module,
) -> Tuple[float, float, float]:
    """
    返回:
      avg_loss, avg_phys_loss, triple_acc
    triple_acc: 三个分类头都预测正确的比例
    """
    model.eval()
    total_loss = 0.0
    total_phys_loss = 0.0
    correct_triple = 0
    total_samples = 0

    with torch.no_grad():
        for x, y_car, y_ped, y_cyc, rho_star, kappa_star in dataloader:
            x = x.to(DEVICE)
            y_car = y_car.to(DEVICE)
            y_ped = y_ped.to(DEVICE)
            y_cyc = y_cyc.to(DEVICE)
            rho_star = rho_star.to(DEVICE)
            kappa_star = kappa_star.to(DEVICE)

            logits_car, logits_ped, logits_cyc, phys_pred = model(x)

            # classification loss
            loss_car = criterion_ce(logits_car, y_car)
            loss_ped = criterion_ce(logits_ped, y_ped)
            loss_cyc = criterion_ce(logits_cyc, y_cyc)
            loss_match = loss_car + loss_ped + loss_cyc

            # physics loss
            rho_hat = phys_pred[:, 0]
            kappa_hat = phys_pred[:, 1]
            loss_rho = criterion_smooth(rho_hat, rho_star)
            loss_kappa = criterion_smooth(kappa_hat, kappa_star)
            loss_phys = loss_rho + loss_kappa

            loss = ALPHA_MATCH * loss_match + BETA_PHYS * loss_phys

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_phys_loss += loss_phys.item() * batch_size
            total_samples += batch_size

            # triple accuracy: 3 heads all correct
            pred_car = logits_car.argmax(dim=1)
            pred_ped = logits_ped.argmax(dim=1)
            pred_cyc = logits_cyc.argmax(dim=1)
            triple_correct = (
                (pred_car == y_car)
                & (pred_ped == y_ped)
                & (pred_cyc == y_cyc)
            )
            correct_triple += triple_correct.sum().item()

    avg_loss = total_loss / total_samples
    avg_phys_loss = total_phys_loss / total_samples
    triple_acc = correct_triple / total_samples
    return avg_loss, avg_phys_loss, triple_acc


# ================== 主函数 ==================


def main():
    # Dataset & Split
    dataset = KittiCaseDataset(DATA_DIR)
    n_total = len(dataset)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False
    )

    # Model
    model = PINNRouterMLP(input_dim=9, embed_dim=EMBED_DIM,
                           num_classes=NUM_CLASSES).to(DEVICE)

    # Loss & Optimizer
    criterion_ce = nn.CrossEntropyLoss()
    criterion_smooth = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print(f"[Info] Device: {DEVICE}")
    print(f"[Info] Train: {n_train}, Val: {n_val}")

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_ce, criterion_smooth
        )
        val_loss, val_phys_loss, val_acc = evaluate(
            model, val_loader, criterion_ce, criterion_smooth
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Phys Loss: {val_phys_loss:.4f} | "
            f"Val Triple Acc: {val_acc:.4f}"
        )

        # 简单的 best 模型保存准则：看 val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"[Save] Best model updated -> {MODEL_SAVE_PATH}")

    print(f"[Done] Training finished. Best Val Loss = {best_val_loss:.4f}")
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[Done] Final model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
