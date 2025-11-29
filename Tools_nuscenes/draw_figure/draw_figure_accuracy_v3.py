

import re
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = "/home/frank/Pu/sci_ML/Tools_nuscenes/training_val.log"
SAVE_PATH = "/home/frank/Pu/sci_ML/Tools_nuscenes/val_acc_comparison.png"
MAX_EPOCHS = 30

acc_no_pibo_xgb = []
acc_pibo_stage0 = []
acc_pibo_xgb_stage1 = []
acc_mlp_stage1 = []

mode = None
stage = None


def parse_val_acc_from_line(line: str):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 2:
        return None
    try:
        val_acc = float(parts[-1])
        return val_acc
    except ValueError:
        return None


# ============================
# Parse log file
# ============================
with open(LOG_PATH, "r") as f:
    for line in f:
        # identify runs
        if "Training_PIBO_XGBoost_v2.py" in line:
            mode, stage = "no_pibo_xgb", None
            continue
        if "Training_PIBO_XGBoost_v5.py" in line:
            mode, stage = "pibo_xgb_v5", None
            continue
        if "/Training.py" in line and "python" in line:
            mode, stage = "mlp", None
            continue

        # identify stages
        if "====================== XGBoost Training" in line:
            stage = "xgb_no_pibo"
            continue
        if "========== Stage 0: PIBO" in line:
            stage = "pibo_stage0" if mode == "pibo_xgb_v5" else None
            continue
        if "====================== Stage 1: Supervised Training" in line:
            if mode == "pibo_xgb_v5":
                stage = "pibo_xgb_stage1"
            elif mode == "mlp":
                stage = "mlp_stage1"
            continue

        # parse data rows
        if "|" in line and line.strip()[0].isdigit():
            val_acc = parse_val_acc_from_line(line)
            if val_acc is None:
                continue

            if stage == "xgb_no_pibo":
                acc_no_pibo_xgb.append(val_acc)
            elif stage == "pibo_stage0":
                acc_pibo_stage0.append(val_acc)
            elif stage == "pibo_xgb_stage1":
                acc_pibo_xgb_stage1.append(val_acc)
            elif stage == "mlp_stage1":
                acc_mlp_stage1.append(val_acc)


# ============================
# Build curves
# ============================
curve_no_pibo = np.array(acc_no_pibo_xgb[:MAX_EPOCHS])
curve_pibo_xgb = np.array((acc_pibo_stage0 + acc_pibo_xgb_stage1)[:MAX_EPOCHS])
curve_pibo_stage0 = np.array(acc_pibo_stage0[:10])
curve_mlp = np.array(acc_mlp_stage1[:MAX_EPOCHS])
epochs = np.arange(1, MAX_EPOCHS + 1)


# ============================
# Plot
# ============================
plt.figure(figsize=(10, 5))

# No PIBO XGBoost
plt.plot(
    epochs[:len(curve_no_pibo)],
    curve_no_pibo,
    marker="o",
    linestyle="-",
    color="gray",
    label="No PIBO (XGBoost) - Val Acc",
)

# PIBO Stage0 + Stage1
plt.plot(
    epochs[:len(curve_pibo_xgb)],
    curve_pibo_xgb,
    marker="s",
    linestyle="-",
    color="#E69F00",
    label="PIBO + XGBoost (Stage0+Stage1) - Val Acc",
)

# PIBO Stage0 only
plt.plot(
    epochs[:len(curve_pibo_stage0)],
    curve_pibo_stage0,
    marker="^",
    linestyle="--",
    color="#E69F00",
    label="PIBO Stage0 (first 10 epochs)",
)

# MLP
plt.plot(
    epochs[:len(curve_mlp)],
    curve_mlp,
    marker="D",
    linestyle="-",
    color="#56B4E9",
    label="PIBO + MLP Stage1 - Val Acc",
)

plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Validation Accuracy", fontsize=18)
plt.title("Epoch-wise Validation Accuracy Comparison (First 30 Epochs)", fontsize=20)
plt.xlim(1, MAX_EPOCHS)
plt.ylim(0.0, 1.02)
plt.grid(True, linestyle="--", alpha=0.4)
# plt.legend(loc="lower right")
plt.legend(loc="right")
plt.tight_layout()

# plt.legend(
#     bbox_to_anchor=(1.02, 1),      # 右侧外部
#     loc="upper left", 
#     borderaxespad=0., 
#     fontsize=14
# )
# plt.tight_layout(rect=[0, 0, 0.85, 1])   # 留出右侧空间


# ============================
# Save PNG
# ============================
plt.savefig(SAVE_PATH, dpi=300)
print(f"[Saved] Figure saved to: {SAVE_PATH}")

plt.show()


