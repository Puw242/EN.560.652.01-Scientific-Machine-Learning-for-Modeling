#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = "/home/frank/Pu/sci_ML/Tools_nuscenes/training_val.log"
MAX_EPOCH = 30  # 只画前 30 个 epoch


def parse_table_block(text_block):
    """
    解析形如:
       1 |    0.7409 |   0.8267 |  0.8333
    的表格，返回 epoch, train_loss, val_loss, val_acc 三个 list。
    """
    epoch_list, train_list, val_list, acc_list = [], [], [], []
    line_pat = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
    )
    for line in text_block.splitlines():
        m = line_pat.match(line)
        if m:
            e = int(m.group(1))
            tr = float(m.group(2))
            vl = float(m.group(3))
            ac = float(m.group(4))
            epoch_list.append(e)
            train_list.append(tr)
            val_list.append(vl)
            acc_list.append(ac)
    return epoch_list, train_list, val_list, acc_list


def parse_xgb_only_val_loss(log_text):
    """
    从 '====================== XGBoost Training ======================'
    部分解析 Val mlogloss（Curve 1）。
    """
    start = log_text.find("====================== XGBoost Training ======================")
    if start == -1:
        return []
    sub = log_text[start:]
    # 表格从 "Iter |" 开始到下一块空行结束
    table_start = sub.find("Iter |")
    if table_start == -1:
        return []
    sub = sub[table_start:]
    table_end = sub.find("\n\n[Result]")  # 到 Result 之前
    if table_end != -1:
        sub = sub[:table_end]

    # 重用表格解析，但列名不同，需要新的正则
    val_losses = []
    line_pat = re.compile(
        r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
    )
    for line in sub.splitlines():
        m = line_pat.match(line)
        if m:
            vl = float(m.group(3))  # 第 3 列是 Val mlogloss
            val_losses.append(vl)
    return val_losses


def parse_stage1_from_block(log_text, stage1_header):
    """
    通用函数：从含有 Stage 1 表格的块中解析 (epoch, train, val, acc)。
    stage1_header 是 Stage 1 标题行的前缀字符串。
    """
    start = log_text.find(stage1_header)
    if start == -1:
        return [], [], [], []
    sub = log_text[start:]
    table_start = sub.find("Epoch |")
    if table_start == -1:
        return [], [], [], []
    sub = sub[table_start:]
    # 截到下一个空行或下一个 '[' 之前
    end_idx = sub.find("\n\n")
    if end_idx != -1:
        sub = sub[:end_idx]

    return parse_table_block(sub)


def main():
    with open(LOG_PATH, "r") as f:
        log_text = f.read()

    # -------- Curve 1: XGBoost only (Val mlogloss) --------
    curve1_val = parse_xgb_only_val_loss(log_text)
    curve1_val = curve1_val[:MAX_EPOCH]
    epochs1 = np.arange(1, len(curve1_val) + 1)

    # -------- Curve 2: PIBO + XGBoost (Stage 1 Val CE) --------
    stage1_header_xgb_pibo = "====================== Stage 1: Supervised Training ======================"
    _, _, curve2_val, _ = parse_stage1_from_block(
        log_text, stage1_header_xgb_pibo
    )
    curve2_val = curve2_val[:MAX_EPOCH]
    epochs2 = np.arange(1, len(curve2_val) + 1)

    # -------- Curve 3: MLP + PIBO (Stage 1 Val CE) --------
    stage1_header_mlp = "====================== Stage 1: Supervised Training ======================\nEpoch |  Train CE (±Phys)"
    # 为了稳一点，这里只用前半截匹配
    stage1_header_mlp_short = "====================== Stage 1: Supervised Training ======================"
    _, _, curve3_val, _ = parse_stage1_from_block(
        log_text, stage1_header_mlp_short
    )
    # log 文件里有两段 “Stage 1: Supervised Training”，
    # 第一段是 XGBoost-PIBO，第二段是 MLP-PIBO；我们取最后一段当 Curve 3。
    if len(curve3_val) > len(curve2_val):
        # 假设前 len(curve2_val) 是 XGB-PIBO，后面是 MLP-PIBO
        curve3_val = curve3_val[len(curve2_val):]
    curve3_val = curve3_val[:MAX_EPOCH]
    epochs3 = np.arange(1, len(curve3_val) + 1)

    # ----------------- Plot -----------------
    plt.figure(figsize=(7, 4))

    # 颜色大致模仿你的示例图
    if len(curve1_val):
        plt.plot(
            epochs1,
            curve1_val,
            label="XGBoost only (Val mlogloss)",
            marker="o",
            linestyle="-",
        )
    if len(curve2_val):
        plt.plot(
            epochs2,
            curve2_val,
            label="PIBO + XGBoost (Stage 1 Val CE)",
            marker="s",
            linestyle="-",
        )
    if len(curve3_val):
        plt.plot(
            epochs3,
            curve3_val,
            label="MLP + PIBO (Stage 1 Val CE)",
            marker="^",
            linestyle="-",
        )

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch-wise Loss Comparison (First 30 Epochs)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = "/home/frank/Pu/sci_ML/Tools_nuscenes/training_curves_first30.png"
    plt.savefig(out_path, dpi=300)
    print(f"[Saved] Figure written to: {out_path}")


if __name__ == "__main__":
    main()
