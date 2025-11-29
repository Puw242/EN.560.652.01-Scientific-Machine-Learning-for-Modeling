#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 training_val.log 中解析三种方法的训练日志，并画出：
1) Val mlogloss 曲线（前 30 个 epoch，log y）
2) Val Accuracy 曲线（前 30 个 epoch）

日志文件包含：
- XGBoost baseline:  ====================== XGBoost Training ======================
- PIBO + XGBoost:   Training_PIBO_XGBoost_v5.py 里的 Stage 1 表格
- MLP + PIBO:       Training.py 里的 Stage 1 表格
"""

import re
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = "/home/frank/Pu/sci_ML/Tools_nuscenes/training_val.log"
MAX_EPOCH = 30  # 只画前 30 个 epoch


def parse_table_block(lines, start_idx):
    """
    从 start_idx 开始，解析下面的表格：
       <epoch> | <train_loss> | <val_loss> | <val_acc>
    直到遇到第一行无法匹配为止。
    返回: dict(epoch, train_loss, val_loss, val_acc)，都是 list[float]
    """
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
    )

    epochs, tl, vl, acc = [], [], [], []
    for i in range(start_idx, len(lines)):
        m = pattern.match(lines[i])
        if not m:
            break
        e = int(m.group(1))
        train_loss = float(m.group(2))
        val_loss = float(m.group(3))
        val_acc = float(m.group(4))
        epochs.append(e)
        tl.append(train_loss)
        vl.append(val_loss)
        acc.append(val_acc)

    return {
        "epoch": np.array(epochs, dtype=int),
        "train_loss": np.array(tl, dtype=float),
        "val_loss": np.array(vl, dtype=float),
        "val_acc": np.array(acc, dtype=float),
    }


def find_first_epoch_line(lines, header_idx):
    """
    从 header_idx 往后找到第一行能够被 parse_table_block 的正则匹配的行 index。
    """
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
    )
    for i in range(header_idx, len(lines)):
        if pattern.match(lines[i]):
            return i
    return None


def main():
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()

    # ---------- 1) XGBoost baseline ----------
    idx_xgb_header = next(
        i for i, l in enumerate(lines)
        if "====================== XGBoost Training ======================" in l
    )
    idx_xgb_start = find_first_epoch_line(lines, idx_xgb_header)
    xgb_base = parse_table_block(lines, idx_xgb_start)

    # ---------- 2) PIBO + XGBoost (Stage 1, v5) ----------
    # 日志里有两段 "====================== Stage 1: Supervised Training ======================"
    # 第一段来自 Training_PIBO_XGBoost_v5.py，我们将其作为 PIBO+XGB
    stage1_indices = [
        i for i, l in enumerate(lines)
        if "====================== Stage 1: Supervised Training ======================" in l
    ]
    if len(stage1_indices) < 2:
        raise RuntimeError("没有找到两段 Stage 1 表格，请检查日志内容。")

    idx_pibo_xgb_header = stage1_indices[0]
    idx_pibo_xgb_start = find_first_epoch_line(lines, idx_pibo_xgb_header)
    pibo_xgb = parse_table_block(lines, idx_pibo_xgb_start)

    # ---------- 3) MLP + PIBO (Stage 1, Training.py) ----------
    idx_mlp_header = stage1_indices[1]
    idx_mlp_start = find_first_epoch_line(lines, idx_mlp_header)
    mlp_pibo = parse_table_block(lines, idx_mlp_start)

    # ---------- 只取前 MAX_EPOCH 个 ----------
    def truncate(curve_dict):
        mask = curve_dict["epoch"] <= MAX_EPOCH
        for k in list(curve_dict.keys()):
            curve_dict[k] = curve_dict[k][mask]

    truncate(xgb_base)
    truncate(pibo_xgb)
    truncate(mlp_pibo)

    # ---------- 画图：Loss ----------
    plt.figure(figsize=(7, 4))
    plt.title("Validation Loss vs. Epoch (Log Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Val mlogloss")
    plt.yscale("log")

    # 颜色大致仿照截图：灰 / 橙 / 蓝
    plt.plot(
        xgb_base["epoch"], xgb_base["val_loss"],
        label="XGBoost only (Curve 1)",
        linestyle="--", marker="o", linewidth=1.5, alpha=0.8, color="#888888"
    )
    plt.plot(
        pibo_xgb["epoch"], pibo_xgb["val_loss"],
        label="PIBO + XGBoost (Curve 2)",
        linestyle="-", marker="s", linewidth=1.8, alpha=0.9, color="#ffb000"
    )
    plt.plot(
        mlp_pibo["epoch"], mlp_pibo["val_loss"],
        label="MLP + PIBO (Curve 3)",
        linestyle="-.", marker="^", linewidth=1.5, alpha=0.9, color="#4477aa"
    )

    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_loss_curves_first30.png", dpi=300)

    # ---------- 画图：Accuracy ----------
    plt.figure(figsize=(7, 4))
    plt.title("Validation Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")

    plt.plot(
        xgb_base["epoch"], xgb_base["val_acc"],
        label="XGBoost only (Curve 1)",
        linestyle="--", marker="o", linewidth=1.5, alpha=0.8, color="#888888"
    )
    plt.plot(
        pibo_xgb["epoch"], pibo_xgb["val_acc"],
        label="PIBO + XGBoost (Curve 2)",
        linestyle="-", marker="s", linewidth=1.8, alpha=0.9, color="#ffb000"
    )
    plt.plot(
        mlp_pibo["epoch"], mlp_pibo["val_acc"],
        label="MLP + PIBO (Curve 3)",
        linestyle="-.", marker="^", linewidth=1.5, alpha=0.9, color="#4477aa"
    )

    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_acc_curves_first30.png", dpi=300)

    print("Saved plots: val_loss_curves_first30.png, val_acc_curves_first30.png")


if __name__ == "__main__":
    main()
