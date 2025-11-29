#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training / validation curves for Sci_ML KITTI case MLP.

根据训练脚本输出的结果手动填入：
[Epoch 01] Train Loss: 3.2227 | Val Loss: 3.2114 | Val Acc: 0.3611
[Epoch 02] Train Loss: 3.1217 | Val Loss: 3.1407 | Val Acc: 0.4444
[Epoch 03] Train Loss: 3.0252 | Val Loss: 3.0810 | Val Acc: 0.5833
[Epoch 04] Train Loss: 2.9280 | Val Loss: 3.0314 | Val Acc: 0.5556
[Epoch 05] Train Loss: 2.8329 | Val Loss: 2.9838 | Val Acc: 0.4167
[Epoch 06] Train Loss: 2.7332 | Val Loss: 2.9419 | Val Acc: 0.4167
[Epoch 07] Train Loss: 2.6362 | Val Loss: 2.9085 | Val Acc: 0.4167
[Epoch 08] Train Loss: 2.5372 | Val Loss: 2.8559 | Val Acc: 0.4167
[Epoch 09] Train Loss: 2.4421 | Val Loss: 2.8140 | Val Acc: 0.4167
[Epoch 10] Train Loss: 2.3409 | Val Loss: 2.7538 | Val Acc: 0.4167
"""

import os
import matplotlib.pyplot as plt

# ================== 手动填入的结果 ==================
epochs = list(range(1, 11))

train_loss = [
    3.2227,
    3.1217,
    3.0252,
    2.9280,
    2.8329,
    2.7332,
    2.6362,
    2.5372,
    2.4421,
    2.3409,
]

val_loss = [
    3.2114,
    3.1407,
    3.0810,
    3.0314,
    2.9838,
    2.9419,
    2.9085,
    2.8559,
    2.8140,
    2.7538,
]

val_acc = [
    0.3611,
    0.4444,
    0.5833,
    0.5556,
    0.4167,
    0.4167,
    0.4167,
    0.4167,
    0.4167,
    0.4167,
]

# ================== Loss 曲线 ==================
plt.figure(figsize=(7, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="s", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Sci-ML KITTI Case MLP: Train/Val Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

loss_fig_path = "kitti_loss_curve.png"
plt.savefig(loss_fig_path, dpi=200)
print(f"[Saved] {os.path.abspath(loss_fig_path)}")

# ================== Val Acc 曲线 ==================
plt.figure(figsize=(7, 5))
plt.plot(epochs, val_acc, marker="^")
plt.xlabel("Epoch")
plt.ylabel("Val Accuracy")
plt.title("Sci-ML KITTI Case MLP: Val Accuracy")
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

acc_fig_path = "kitti_val_acc.png"
plt.savefig(acc_fig_path, dpi=200)
print(f"[Saved] {os.path.abspath(acc_fig_path)}")

plt.show()
