#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training / validation curves for Sci_ML KITTI case MLP (20 epochs).
"""

import os
import matplotlib.pyplot as plt

# ================== 手动填入的 20 轮结果 ==================

epochs = list(range(1, 21))

train_loss = [
    3.2227, 3.1217, 3.0252, 2.9280, 2.8329,
    2.7332, 2.6362, 2.5372, 2.4421, 2.3409,
    2.2480, 2.1572, 2.0518, 1.9260, 1.8049,
    1.6712, 1.5477, 1.4147, 1.2958, 1.1888
]

val_loss = [
    3.2114, 3.1407, 3.0810, 3.0314, 2.9838,
    2.9419, 2.9085, 2.8559, 2.8140, 2.7538,
    2.6956, 2.6118, 2.4594, 2.3601, 2.2651,
    2.1478, 2.0240, 1.9720, 1.9405, 1.9440
]

val_acc = [
    0.3611, 0.4444, 0.5833, 0.5556, 0.4167,
    0.4167, 0.4167, 0.4167, 0.4167, 0.4167,
    0.4167, 0.4167, 0.5000, 0.5000, 0.5000,
    0.5000, 0.5278, 0.5556, 0.5833, 0.6111
]

# ================== Loss 曲线 ==================
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="s", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("KITTI Sci-ML MLP Training: Loss Curves (20 epochs)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

loss_fig_path = "kitti_loss_curve_20epoch.png"
plt.savefig(loss_fig_path, dpi=200)
print(f"[Saved] {os.path.abspath(loss_fig_path)}")

# ================== Val Acc 曲线 ==================
plt.figure(figsize=(8, 6))
plt.plot(epochs, val_acc, marker="^")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("KITTI Sci-ML MLP: Validation Accuracy (20 epochs)")
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

acc_fig_path = "kitti_val_acc_20epoch.png"
plt.savefig(acc_fig_path, dpi=200)
print(f"[Saved] {os.path.abspath(acc_fig_path)}")

plt.show()
