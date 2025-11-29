#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import matplotlib.pyplot as plt

# ---------- 原始日志：Curve 1（精简数据集，Baseline MLP） ----------
log_curve1 = r"""
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
[Epoch 11] Train Loss: 2.2480 | Val Loss: 2.6956 | Val Acc: 0.4167
[Epoch 12] Train Loss: 2.1572 | Val Loss: 2.6118 | Val Acc: 0.4167
[Epoch 13] Train Loss: 2.0518 | Val Loss: 2.4594 | Val Acc: 0.5000
[Epoch 14] Train Loss: 1.9260 | Val Loss: 2.3601 | Val Acc: 0.5000
[Epoch 15] Train Loss: 1.8049 | Val Loss: 2.2651 | Val Acc: 0.5000
[Epoch 16] Train Loss: 1.6712 | Val Loss: 2.1478 | Val Acc: 0.5000
[Epoch 17] Train Loss: 1.5477 | Val Loss: 2.0240 | Val Acc: 0.5278
[Epoch 18] Train Loss: 1.4147 | Val Loss: 1.9720 | Val Acc: 0.5556
[Epoch 19] Train Loss: 1.2958 | Val Loss: 1.9405 | Val Acc: 0.5833
[Epoch 20] Train Loss: 1.1888 | Val Loss: 1.9440 | Val Acc: 0.6111
[Epoch 21] Train Loss: 1.1003 | Val Loss: 1.9926 | Val Acc: 0.6111
[Epoch 22] Train Loss: 1.0173 | Val Loss: 1.9773 | Val Acc: 0.6389
[Epoch 23] Train Loss: 0.9506 | Val Loss: 1.9969 | Val Acc: 0.6667
[Epoch 24] Train Loss: 0.8850 | Val Loss: 1.9789 | Val Acc: 0.6667
[Epoch 25] Train Loss: 0.8274 | Val Loss: 1.9744 | Val Acc: 0.6667
[Epoch 26] Train Loss: 0.7845 | Val Loss: 1.9583 | Val Acc: 0.6667
[Epoch 27] Train Loss: 0.7370 | Val Loss: 1.9230 | Val Acc: 0.6944
[Epoch 28] Train Loss: 0.6906 | Val Loss: 1.8616 | Val Acc: 0.6944
[Epoch 29] Train Loss: 0.6604 | Val Loss: 1.8841 | Val Acc: 0.6944
[Epoch 30] Train Loss: 0.6248 | Val Loss: 1.7871 | Val Acc: 0.6944
"""

# ---------- Curve 2：第一次 PIBO + 两阶段训练（30 epoch） ----------
log_curve2_stage0 = r"""
Epoch |  Train L_phys  |  Val CE (proxy)  |  Val Acc (proxy)
---------------------------------------------------------------
    1 |        1.0419 |          3.2709 |      0.1944
    2 |        0.9883 |          3.2475 |      0.3889
    3 |        0.9553 |          3.2308 |      0.4167
    4 |        0.8977 |          3.2160 |      0.4167
    5 |        0.8508 |          3.2094 |      0.3889
"""

log_curve2_stage1 = r"""
Epoch |  Train CE (±Phys) |   Val CE       |  Val Acc
------------------------------------------------------
    1 |           3.1642 |       3.1651 |    0.4167
    2 |           3.0960 |       3.1007 |    0.4444
    3 |           3.0078 |       3.0158 |    0.4722
    4 |           2.8952 |       2.9396 |    0.4722
    5 |           2.7850 |       2.8568 |    0.5833
    6 |           2.6682 |       2.7784 |    0.5833
    7 |           2.5525 |       2.6950 |    0.5278
    8 |           2.4217 |       2.5983 |    0.5278
    9 |           2.3014 |       2.5390 |    0.5000
   10 |           2.1848 |       2.4841 |    0.5000
   11 |           2.0610 |       2.4078 |    0.5000
   12 |           1.9418 |       2.2858 |    0.5000
   13 |           1.8152 |       2.1837 |    0.5000
   14 |           1.6830 |       2.0721 |    0.5278
   15 |           1.5478 |       1.9736 |    0.5556
   16 |           1.4137 |       1.9208 |    0.5556
   17 |           1.2692 |       1.8266 |    0.6111
   18 |           1.1489 |       1.8022 |    0.6111
   19 |           1.0314 |       1.8061 |    0.6389
   20 |           0.9321 |       1.8422 |    0.6389
   21 |           0.8576 |       1.8673 |    0.6667
   22 |           0.7878 |       1.8788 |    0.6667
   23 |           0.7261 |       1.8461 |    0.6944
   24 |           0.6848 |       1.9040 |    0.6944
   25 |           0.6414 |       1.8104 |    0.6944
   26 |           0.6021 |       1.7438 |    0.6944
   27 |           0.5725 |       1.7522 |    0.6944
   28 |           0.5414 |       1.6946 |    0.6944
   29 |           0.5157 |       1.6270 |    0.7222
   30 |           0.4958 |       1.5669 |    0.7222
"""

# ---------- Curve 3：最终 PIBO 两阶段训练（50 epoch） ----------
log_curve3_stage0 = log_curve2_stage0  # 同样的 Stage 0

log_curve3_stage1 = r"""
Epoch |  Train CE (±Phys) |   Val CE       |  Val Acc
------------------------------------------------------
    1 |           3.1642 |       3.1651 |    0.4167
    2 |           3.0960 |       3.1007 |    0.4444
    3 |           3.0078 |       3.0158 |    0.4722
    4 |           2.8952 |       2.9396 |    0.4722
    5 |           2.7850 |       2.8568 |    0.5833
    6 |           2.6682 |       2.7784 |    0.5833
    7 |           2.5525 |       2.6950 |    0.5278
    8 |           2.4217 |       2.5983 |    0.5278
    9 |           2.3014 |       2.5390 |    0.5000
   10 |           2.1848 |       2.4841 |    0.5000
   11 |           2.0610 |       2.4078 |    0.5000
   12 |           1.9418 |       2.2858 |    0.5000
   13 |           1.8152 |       2.1837 |    0.5000
   14 |           1.6830 |       2.0721 |    0.5278
   15 |           1.5478 |       1.9736 |    0.5556
   16 |           1.4137 |       1.9208 |    0.5556
   17 |           1.2692 |       1.8266 |    0.6111
   18 |           1.1489 |       1.8022 |    0.6111
   19 |           1.0314 |       1.8061 |    0.6389
   20 |           0.9321 |       1.8422 |    0.6389
   21 |           0.8576 |       1.8673 |    0.6667
   22 |           0.7878 |       1.8788 |    0.6667
   23 |           0.7261 |       1.8461 |    0.6944
   24 |           0.6848 |       1.9040 |    0.6944
   25 |           0.6414 |       1.8104 |    0.6944
   26 |           0.6021 |       1.7438 |    0.6944
   27 |           0.5725 |       1.7522 |    0.6944
   28 |           0.5414 |       1.6946 |    0.6944
   29 |           0.5157 |       1.6270 |    0.7222
   30 |           0.4958 |       1.5669 |    0.7222
   31 |           0.4791 |       1.4863 |    0.7222
   32 |           0.4606 |       1.4363 |    0.7222
   33 |           0.4508 |       1.5050 |    0.7222
   34 |           0.4454 |       1.3284 |    0.7222
   35 |           0.4214 |       1.3593 |    0.7222
   36 |           0.3998 |       1.2811 |    0.7222
   37 |           0.3915 |       1.1790 |    0.7222
   38 |           0.3827 |       1.2063 |    0.7222
   39 |           0.3712 |       1.1661 |    0.7500
   40 |           0.3534 |       1.0470 |    0.8333
   41 |           0.3602 |       0.9298 |    0.8333
   42 |           0.3525 |       1.0943 |    0.7778
   43 |           0.3389 |       1.0904 |    0.7778
   44 |           0.3198 |       0.9727 |    0.8889
   45 |           0.3090 |       0.8441 |    0.9444
   46 |           0.3101 |       0.8472 |    0.9444
   47 |           0.3021 |       0.9224 |    0.9167
   48 |           0.2926 |       0.9391 |    0.8611
   49 |           0.2883 |       0.8293 |    0.9444
   50 |           0.2846 |       0.8849 |    0.8889
"""

# ---------- Curve 4：全数据集 MLP ----------
log_curve4 = r"""
[Epoch 01] Train Loss: 3.2250 | Val Loss: 3.2727 | Val Acc: 0.2500
[Epoch 02] Train Loss: 3.1939 | Val Loss: 3.2657 | Val Acc: 0.2500
[Epoch 03] Train Loss: 3.1626 | Val Loss: 3.2618 | Val Acc: 0.2500
[Epoch 04] Train Loss: 3.1356 | Val Loss: 3.2625 | Val Acc: 0.2500
[Epoch 05] Train Loss: 3.1054 | Val Loss: 3.2641 | Val Acc: 0.4167
[Epoch 06] Train Loss: 3.0773 | Val Loss: 3.2670 | Val Acc: 0.4167
[Epoch 07] Train Loss: 3.0458 | Val Loss: 3.2686 | Val Acc: 0.3333
[Epoch 08] Train Loss: 3.0090 | Val Loss: 3.2682 | Val Acc: 0.3333
[Epoch 09] Train Loss: 2.9742 | Val Loss: 3.2713 | Val Acc: 0.3333
[Epoch 10] Train Loss: 2.9325 | Val Loss: 3.2737 | Val Acc: 0.3333
[Epoch 11] Train Loss: 2.8886 | Val Loss: 3.2752 | Val Acc: 0.3333
[Epoch 12] Train Loss: 2.8429 | Val Loss: 3.2796 | Val Acc: 0.3333
[Epoch 13] Train Loss: 2.7829 | Val Loss: 3.2776 | Val Acc: 0.3333
[Epoch 14] Train Loss: 2.7338 | Val Loss: 3.2908 | Val Acc: 0.3333
[Epoch 15] Train Loss: 2.6765 | Val Loss: 3.3111 | Val Acc: 0.3333
[Epoch 16] Train Loss: 2.6199 | Val Loss: 3.3332 | Val Acc: 0.3333
[Epoch 17] Train Loss: 2.5626 | Val Loss: 3.3544 | Val Acc: 0.3333
[Epoch 18] Train Loss: 2.5202 | Val Loss: 3.3906 | Val Acc: 0.3333
[Epoch 19] Train Loss: 2.4825 | Val Loss: 3.4238 | Val Acc: 0.3333
[Epoch 20] Train Loss: 2.4570 | Val Loss: 3.4694 | Val Acc: 0.3333
[Epoch 21] Train Loss: 2.4395 | Val Loss: 3.5185 | Val Acc: 0.3333
[Epoch 22] Train Loss: 2.4213 | Val Loss: 3.5318 | Val Acc: 0.3333
[Epoch 23] Train Loss: 2.4077 | Val Loss: 3.5248 | Val Acc: 0.3333
[Epoch 24] Train Loss: 2.3955 | Val Loss: 3.5136 | Val Acc: 0.3333
[Epoch 25] Train Loss: 2.3842 | Val Loss: 3.4868 | Val Acc: 0.3333
[Epoch 26] Train Loss: 2.3729 | Val Loss: 3.4507 | Val Acc: 0.3333
[Epoch 27] Train Loss: 2.3597 | Val Loss: 3.4180 | Val Acc: 0.3333
[Epoch 28] Train Loss: 2.3497 | Val Loss: 3.3817 | Val Acc: 0.3333
[Epoch 29] Train Loss: 2.3403 | Val Loss: 3.3526 | Val Acc: 0.3333
[Epoch 30] Train Loss: 2.3333 | Val Loss: 3.3081 | Val Acc: 0.3333
"""

# ---------- 解析函数 ----------

def parse_epoch_val_from_simple_log(text):
    """解析 [Epoch xx] 行，返回 epochs, val_loss, val_acc"""
    pattern = re.compile(
        r"\[Epoch\s+(\d+)\].*?Val Loss:\s*([\d\.]+)\s*\|\s*Val Acc:\s*([\d\.]+)"
    )
    epochs, vloss, vacc = [], [], []
    for m in pattern.finditer(text):
        epochs.append(int(m.group(1)))
        vloss.append(float(m.group(2)))
        vacc.append(float(m.group(3)))
    return epochs, vloss, vacc


def parse_epoch_val_from_pibo_block(text):
    """
    解析 PIBO 风格的表格行：
       1 |   ...  | Val CE | Val Acc
    返回 val_loss(list), val_acc(list)
    """
    pattern = re.compile(
        r"^\s*\d+\s*\|\s*[\d\.]+\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)",
        re.MULTILINE,
    )
    vloss, vacc = [], []
    for m in pattern.finditer(text):
        vloss.append(float(m.group(1)))
        vacc.append(float(m.group(2)))
    return vloss, vacc


# ---------- 生成四条曲线的数据 ----------

# Curve 1
e1, vloss1, vacc1 = parse_epoch_val_from_simple_log(log_curve1)

# Curve 2 = Stage0(5) + Stage1(30)
vl_s0_c2, va_s0_c2 = parse_epoch_val_from_pibo_block(log_curve2_stage0)
vl_s1_c2, va_s1_c2 = parse_epoch_val_from_pibo_block(log_curve2_stage1)
vloss2 = vl_s0_c2 + vl_s1_c2
vacc2 = va_s0_c2 + va_s1_c2
e2 = list(range(1, len(vloss2) + 1))

# Curve 3 = Stage0(5) + Stage1(50)
vl_s0_c3, va_s0_c3 = parse_epoch_val_from_pibo_block(log_curve3_stage0)
vl_s1_c3, va_s1_c3 = parse_epoch_val_from_pibo_block(log_curve3_stage1)
vloss3 = vl_s0_c3 + vl_s1_c3
vacc3 = va_s0_c3 + va_s1_c3
e3 = list(range(1, len(vloss3) + 1))

# Curve 4
e4, vloss4, vacc4 = parse_epoch_val_from_simple_log(log_curve4)

MAX_EPOCH = max(e1[-1], e2[-1], e3[-1], e4[-1])

# ---------- 画图设置 ----------
plt.rcParams["font.size"] = 16  # 整体字号大一点

# # ========== 图 1：Validation Loss ==========
# fig1, ax1 = plt.subplots(figsize=(10, 5))

# ax1.plot(e1, vloss1, marker="o", linestyle="-", label="Curve 1: MLP (filtered 60)")
# ax1.plot(e2, vloss2, marker="s", linestyle="-.", label="Curve 2: PIBO+MLP (30 epochs)")
# ax1.plot(e3, vloss3, marker="^", linestyle="--", label="Curve 3: PIBO+MLP (50 epochs)")
# ax1.plot(e4, vloss4, marker="d", linestyle=":", label="Curve 4: MLP (full dataset)")

# ax1.set_xlabel("Epoch")
# ax1.set_ylabel("Validation Loss")
# ax1.set_title("Validation Loss Curves")
# ax1.set_xlim(1, MAX_EPOCH)
# ax1.grid(True, linestyle="--", alpha=0.4)
# ax1.legend(loc="best")
# fig1.tight_layout()
# fig1.savefig("kitti_val_loss_curves.png", dpi=300)

# # ========== 图 2：Validation Accuracy ==========
# fig2, ax2 = plt.subplots(figsize=(10, 5))

# ax2.plot(e1, vacc1, marker="o", linestyle="-", label="Curve 1: MLP (filtered 60)")
# ax2.plot(e2, vacc2, marker="s", linestyle="-.", label="Curve 2: PIBO+MLP (30 epochs)")
# ax2.plot(e3, vacc3, marker="^", linestyle="--", label="Curve 3: PIBO+MLP (50 epochs)")
# ax2.plot(e4, vacc4, marker="d", linestyle=":", label="Curve 4: MLP (full dataset)")

# ax2.set_xlabel("Epoch")
# ax2.set_ylabel("Validation Accuracy")
# ax2.set_title("Validation Accuracy Curves")
# ax2.set_xlim(1, MAX_EPOCH)
# ax2.set_ylim(0.0, 1.05)
# ax2.grid(True, linestyle="--", alpha=0.4)
# ax2.legend(loc="best")
# fig2.tight_layout()
# fig2.savefig("kitti_val_acc_curves.png", dpi=300)

# print("Saved: kitti_val_loss_curves.png, kitti_val_acc_curves.png")


# ========== 图 1：Validation Loss ==========
fig1, ax1 = plt.subplots(figsize=(10, 5))

# Curve 1: baseline 60 filtered MLP  → gray
ax1.plot(
    e1, vloss1,
    marker="o", linestyle="-",
    color="gray",
    linewidth=2,
    label="Curve 1: MLP (filtered 60)"
)

# Curve 2: PIBO+MLP 30 epoch → gold (#E69F00)
ax1.plot(
    e2, vloss2,
    marker="s", linestyle="-",
    color="#E69F00",
    linewidth=2,
    label="Curve 2: PIBO + MLP (30 epochs)"
)

# Curve 3: PIBO+MLP 50 epoch → gold (#E69F00)
ax1.plot(
    e3, vloss3,
    marker="s", linestyle="-",
    color="#E69F00",
    linewidth=2,
    alpha=0.6,   # 可以略微区分
    label="Curve 3: PIBO + MLP (50 epochs)"
)

# Curve 4: full dataset MLP → blue (#56B4E9)
ax1.plot(
    e4, vloss4,
    marker="D", linestyle="-",
    color="#56B4E9",
    linewidth=2,
    label="Curve 4: MLP (full dataset)"
)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation Loss")
ax1.set_title("Validation Loss Curves")
ax1.set_xlim(1, MAX_EPOCH)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(loc="best")
fig1.tight_layout()
fig1.savefig("kitti_val_loss_curves.png", dpi=300)


# ========== 图 2：Validation Accuracy ==========
fig2, ax2 = plt.subplots(figsize=(10, 5))

# Curve 1 → gray
ax2.plot(
    e1, vacc1,
    marker="o", linestyle="-",
    color="gray",
    linewidth=2,
    label="Curve 1: MLP (filtered 60)"
)

# Curve 2 → gold
ax2.plot(
    e2, vacc2,
    marker="s", linestyle="-",
    color="#E69F00",
    linewidth=2,
    label="Curve 2: PIBO + MLP (30 epochs)"
)

# Curve 3 → gold
ax2.plot(
    e3, vacc3,
    marker="s", linestyle="-",
    color="#E69F00",
    linewidth=2,
    alpha=0.6,
    label="Curve 3: PIBO + MLP (50 epochs)"
)

# Curve 4 → blue
ax2.plot(
    e4, vacc4,
    marker="D", linestyle="-",
    color="#56B4E9",
    linewidth=2,
    label="Curve 4: MLP (full dataset)"
)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Accuracy")
ax2.set_title("Validation Accuracy Curves")
ax2.set_xlim(1, MAX_EPOCH)
ax2.set_ylim(0.0, 1.05)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="best")
fig2.tight_layout()
fig2.savefig("kitti_val_acc_curves.png", dpi=300)

print("Saved: kitti_val_loss_curves.png, kitti_val_acc_curves.png")

