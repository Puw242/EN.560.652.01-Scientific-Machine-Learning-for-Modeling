#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = "/home/frank/Pu/sci_ML/Tools_nuscenes/training_val.log"
MAX_EPOCH = 30  # 只画前 30 个 epoch

# -------------------------------------------------------
# 基础解析函数：解析一个 “表格 block”（Epoch | Train | Val | Acc）
# -------------------------------------------------------
def parse_epoch_table(lines, start_idx):
    """
    从 start_idx 之后解析 epoch 表格，直到遇到空行或非表格行为止。
    返回：
      epochs, train_vals, val_vals, acc_vals  （都是 list[float]）
      end_idx: 解析结束行的下标（方便后续继续）
    """
    header_pattern = re.compile(r"^\s*Epoch\s*\|")
    sep_pattern = re.compile(r"^-{5,}")
    row_pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
    )

    # 跳过 header 行和分隔线
    idx = start_idx
    while idx < len(lines) and not header_pattern.search(lines[idx]):
        idx += 1
    idx += 1  # 跳到分隔线
    while idx < len(lines) and sep_pattern.search(lines[idx]):
        idx += 1

    epochs, train_vals, val_vals, acc_vals = [], [], [], []

    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            break
        m = row_pattern.match(line)
        if not m:
            break
        ep = int(m.group(1))
        tr = float(m.group(2))
        va = float(m.group(3))
        ac = float(m.group(4))
        epochs.append(ep)
        train_vals.append(tr)
        val_vals.append(va)
        acc_vals.append(ac)
        idx += 1

    return epochs, train_vals, val_vals, acc_vals, idx


# -------------------------------------------------------
# 解析整体 log：找到 XGBoost-only、PIBO+XGB、MLP+PIBO 三个实验
# -------------------------------------------------------
def parse_all_experiments(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    # 1) 纯 XGBoost（no PIBO）——“====================== XGBoost Training ======================” block
    xgb_only_epochs = []
    xgb_only_train = []
    xgb_only_val = []
    xgb_only_acc = []

    # 纯 XGBoost 表格行格式：
    # "   1 |   1.0546 |   1.0625 |   0.7500"
    xgb_header_pat = re.compile(r"^=+ XGBoost Training =+")
    xgb_row_pat = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
    )
    idx = 0
    while idx < len(lines):
        if xgb_header_pat.search(lines[idx]):
            idx += 2  # 跳过 header + 分隔线
            while idx < len(lines):
                line = lines[idx]
                if not line.strip():
                    break
                m = xgb_row_pat.match(line)
                if not m:
                    break
                ep = int(m.group(1))
                tr = float(m.group(2))
                va = float(m.group(3))
                ac = float(m.group(4))
                xgb_only_epochs.append(ep)
                xgb_only_train.append(tr)
                xgb_only_val.append(va)
                xgb_only_acc.append(ac)
                idx += 1
        else:
            idx += 1

    # 2) 所有 Stage 0 PIBO block（MLP + XGB 两种），用长度区分：len=5 是 MLP，len=10 是 PIBO+XGB
    stage0_pat = re.compile(r"^=+ Stage 0: PIBO \(Physical-Information-Based Initialization\) =+")
    stage0_blocks = []
    idx = 0
    while idx < len(lines):
        if stage0_pat.search(lines[idx]):
            e, tr, va, ac, new_idx = parse_epoch_table(lines, idx + 1)
            stage0_blocks.append((e, tr, va, ac))
            idx = new_idx
        else:
            idx += 1

    # 选出 len=10 的那个作为 PIBO+XGB 的 Stage 0
    stage0_pibo_xgb = None
    stage0_mlp = None
    for e, tr, va, ac in stage0_blocks:
        if len(e) >= 10:
            stage0_pibo_xgb = (e, tr, va, ac)
        elif len(e) == 5:
            stage0_mlp = (e, tr, va, ac)

    # 3) 所有 Stage 1 block，两种：MLP (len≈50)，PIBO+XGB (len≈40)
    stage1_pat = re.compile(r"^=+ Stage 1: Supervised Training =+")
    stage1_blocks = []
    idx = 0
    while idx < len(lines):
        if stage1_pat.search(lines[idx]):
            e, tr, va, ac, new_idx = parse_epoch_table(lines, idx + 1)
            stage1_blocks.append((e, tr, va, ac))
            idx = new_idx
        else:
            idx += 1

    stage1_pibo_xgb = None
    stage1_mlp = None
    for e, tr, va, ac in stage1_blocks:
        if len(e) >= 40 and len(e) <= 45:
            stage1_pibo_xgb = (e, tr, va, ac)
        elif len(e) >= 45:  # 50 个 epoch 左右
            stage1_mlp = (e, tr, va, ac)

    return {
        "xgb_only": {
            "epochs": np.array(xgb_only_epochs, dtype=int),
            "train": np.array(xgb_only_train),
            "val": np.array(xgb_only_val),
            "acc": np.array(xgb_only_acc),
        },
        "stage0_pibo_xgb": None if stage0_pibo_xgb is None else {
            "epochs": np.array(stage0_pibo_xgb[0], dtype=int),
            "train": np.array(stage0_pibo_xgb[1]),
            "val": np.array(stage0_pibo_xgb[2]),
            "acc": np.array(stage0_pibo_xgb[3]),
        },
        "stage1_pibo_xgb": None if stage1_pibo_xgb is None else {
            "epochs": np.array(stage1_pibo_xgb[0], dtype=int),
            "train": np.array(stage1_pibo_xgb[1]),
            "val": np.array(stage1_pibo_xgb[2]),
            "acc": np.array(stage1_pibo_xgb[3]),
        },
        "stage0_mlp": None if stage0_mlp is None else {
            "epochs": np.array(stage0_mlp[0], dtype=int),
            "train": np.array(stage0_mlp[1]),
            "val": np.array(stage0_mlp[2]),
            "acc": np.array(stage0_mlp[3]),
        },
        "stage1_mlp": None if stage1_mlp is None else {
            "epochs": np.array(stage1_mlp[0], dtype=int),
            "train": np.array(stage1_mlp[1]),
            "val": np.array(stage1_mlp[2]),
            "acc": np.array(stage1_mlp[3]),
        },
    }


def main():
    data = parse_all_experiments(LOG_PATH)

    xgb = data["xgb_only"]
    s0_pibo_xgb = data["stage0_pibo_xgb"]
    s1_pibo_xgb = data["stage1_pibo_xgb"]
    s1_mlp = data["stage1_mlp"]

    # 截取前 30 个 epoch
    def first_n(arr, n):
        return arr[: min(len(arr), n)]

    # -------------------- 图 1：Val Loss (log y) --------------------
    plt.figure(figsize=(7, 4))

    # Curve 1：纯 XGBoost
    plt.plot(
        first_n(xgb["epochs"], MAX_EPOCH),
        first_n(xgb["val"], MAX_EPOCH),
        label="No PIBO (XGBoost) - Val loss",
        color="#808080",
        linestyle="-",
        marker="o",
        linewidth=1.5,
        markersize=4,
    )

    # Curve 2：PIBO + XGBoost Stage 1（整体 1~30）
    if s1_pibo_xgb is not None:
        plt.plot(
            first_n(s1_pibo_xgb["epochs"], MAX_EPOCH),
            first_n(s1_pibo_xgb["val"], MAX_EPOCH),
            label="PIBO + XGBoost Stage1 - Val CE",
            color="#E69F00",  # 橙色
            linestyle="-",
            marker="s",
            linewidth=1.5,
            markersize=4,
        )

    # Curve 2 的前 10 个 epoch：用 Stage 0 的 Val CE（颜色不同标记）
    if s0_pibo_xgb is not None:
        plt.plot(
            s0_pibo_xgb["epochs"],  # 1~10
            s0_pibo_xgb["val"],
            label="PIBO Stage0 (first 10 epochs) - Val CE",
            color="#E69F00",
            linestyle="--",
            marker="^",
            linewidth=1.2,
            markersize=5,
        )

    # Curve 3：MLP + PIBO Stage1
    if s1_mlp is not None:
        plt.plot(
            first_n(s1_mlp["epochs"], MAX_EPOCH),
            first_n(s1_mlp["val"], MAX_EPOCH),
            label="PIBO + MLP Stage1 - Val CE",
            color="#56B4E9",  # 蓝色
            linestyle="-",
            marker="d",
            linewidth=1.5,
            markersize=4,
        )

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (mlogloss / CE)")
    plt.title("Epoch-wise Validation Loss Comparison (First 30 Epochs)")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curves_first30.png", dpi=300)

    # -------------------- 图 2：Val Accuracy --------------------
    plt.figure(figsize=(7, 4))

    plt.plot(
        first_n(xgb["epochs"], MAX_EPOCH),
        first_n(xgb["acc"], MAX_EPOCH),
        label="No PIBO (XGBoost) - Val Acc",
        color="#808080",
        linestyle="-",
        marker="o",
        linewidth=1.5,
        markersize=4,
    )

    if s1_pibo_xgb is not None:
        plt.plot(
            first_n(s1_pibo_xgb["epochs"], MAX_EPOCH),
            first_n(s1_pibo_xgb["acc"], MAX_EPOCH),
            label="PIBO + XGBoost Stage1 - Val Acc",
            color="#E69F00",
            linestyle="-",
            marker="s",
            linewidth=1.5,
            markersize=4,
        )

    if s0_pibo_xgb is not None:
        plt.plot(
            s0_pibo_xgb["epochs"],
            s0_pibo_xgb["acc"],
            label="PIBO Stage0 (first 10 epochs) - Val Acc",
            color="#E69F00",
            linestyle="--",
            marker="^",
            linewidth=1.2,
            markersize=5,
        )

    if s1_mlp is not None:
        plt.plot(
            first_n(s1_mlp["epochs"], MAX_EPOCH),
            first_n(s1_mlp["acc"], MAX_EPOCH),
            label="PIBO + MLP Stage1 - Val Acc",
            color="#56B4E9",
            linestyle="-",
            marker="d",
            linewidth=1.5,
            markersize=4,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.title("Epoch-wise Validation Accuracy Comparison (First 30 Epochs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc_curves_first30.png", dpi=300)

    print("Saved: loss_curves_first30.png, acc_curves_first30.png")

if __name__ == "__main__":
    main()
