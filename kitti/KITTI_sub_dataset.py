#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count all (Car_model, Pedestrian_model, Cyclist_model) combos per frame.
If a frame has no valid detection for a class (after filtering by IoU), use 'none' for that class.
Sum of all combo counts equals total #frames (union of frame_ids from 3 files).

运行：
    python /home/frank/Pu/sci_ML/kitti/KITTI_sub_dataset.py
"""

# ================== 路径（按需修改） ==================
CAR_JSONL = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Car.jsonl"
PED_JSONL = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Pedestrian.jsonl"
CYC_JSONL = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Cyclist.jsonl"

OUTPUT_COUNTS_CSV      = "/home/frank/Pu/sci_ML/kitti/kitti_triplet_combo_counts.csv"
OUTPUT_PER_FRAME_CSV   = "/home/frank/Pu/sci_ML/kitti/kitti_triplet_combos_per_frame.csv"
TOP_SHOW = 20
# =====================================================

import os
import sys
import json
import pandas as pd

REQUIRED_KEYS = {
    "frame_id", "class_name", "iou_thr", "iou",
    "is_tp", "ignored", "score", "_selected_model"
}

def read_jsonl(path):
    if not os.path.exists(path):
        sys.exit(f"[Error] File not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                sys.exit(f"[Error] JSON decode failed at {path}:{ln} -> {e}")
    if not rows:
        sys.exit(f"[Error] Empty file: {path}")
    return rows

def load_df(path, expect_class):
    data = read_jsonl(path)
    bad = [i for i, r in enumerate(data) if not REQUIRED_KEYS.issubset(r.keys())]
    if bad:
        sys.exit(f"[Error] Missing keys in {path}, example row idx={bad[0]}.\n"
                 f"Required: {sorted(REQUIRED_KEYS)}")
    df = pd.DataFrame(data)
    # 只保留该类；保留全量帧号用于 union
    all_frames = set(df["frame_id"].astype(str))
    df = df[df["class_name"] == expect_class].copy()
    if df.empty:
        # 该类没有任何记录，也要返回空表和帧集合
        return pd.DataFrame(columns=["frame_id", f"{expect_class}_model", f"{expect_class}_score"]), all_frames

    df["frame_id"] = df["frame_id"].astype(str)
    df["iou_thr"] = pd.to_numeric(df["iou_thr"], errors="coerce")
    df["iou"] = pd.to_numeric(df["iou"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["_selected_model"] = df["_selected_model"].astype(str).str.strip().str.lower()
    df["is_tp"] = df["is_tp"].astype(bool)
    df["ignored"] = df["ignored"].astype(bool)

    # 过滤：严格使用每条记录的 iou_thr
    before = len(df)
    df = df[(df["iou"] >= df["iou_thr"]) & (df["is_tp"]) & (~df["ignored"])].copy()
    after = len(df)
    print(f"[Info] {expect_class}: rows before={before}, after_filter={after}, file={os.path.basename(path)}")

    if df.empty:
        # 返回空的 per-class 结果；后续会用 none 填充
        return pd.DataFrame(columns=["frame_id", f"{expect_class}_model", f"{expect_class}_score"]), all_frames

    # 每帧保留最高分
    df = df.sort_values(["frame_id", "score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["frame_id"], keep="first")

    keep = df[["frame_id", "_selected_model", "score"]].copy()
    keep.rename(columns={
        "_selected_model": f"{expect_class}_model",
        "score": f"{expect_class}_score"
    }, inplace=True)
    return keep, all_frames

def main():
    os.makedirs(os.path.dirname(OUTPUT_COUNTS_CSV), exist_ok=True)

    car_df, frames_car = load_df(CAR_JSONL, "Car")
    ped_df, frames_ped = load_df(PED_JSONL, "Pedestrian")
    cyc_df, frames_cyc = load_df(CYC_JSONL, "Cyclist")

    # 帧号取并集（只要三个文件任一出现过该帧，就统计）
    all_frames = sorted(frames_car | frames_ped | frames_cyc)
    if not all_frames:
        sys.exit("[Error] No frame_id found in any input.")

    # 生成包含所有帧的骨架
    base = pd.DataFrame({"frame_id": all_frames})

    # 外连接合并三个类
    merged = base.merge(car_df, on="frame_id", how="left") \
                 .merge(ped_df, on="frame_id", how="left") \
                 .merge(cyc_df, on="frame_id", how="left")

    # 用 none 填充缺失的模型名
    for col in ["Car_model", "Pedestrian_model", "Cyclist_model"]:
        if col not in merged.columns:
            merged[col] = "none"
        else:
            merged[col] = merged[col].fillna("none").astype(str).str.strip().str.lower()

    # （可选）分数字段保持原样，缺失即 NaN
    # 保存每帧的组合明细
    merged_out_cols = ["frame_id",
                       "Car_model", "Car_score",
                       "Pedestrian_model", "Pedestrian_score",
                       "Cyclist_model", "Cyclist_score"]
    # 某些列可能不存在（例如某类完全没有有效记录），先补齐列
    for c in merged_out_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged[merged_out_cols].to_csv(OUTPUT_PER_FRAME_CSV, index=False)

    # 统计组合频次（包含 none）
    counts = (merged
              .groupby(["Car_model", "Pedestrian_model", "Cyclist_model"], dropna=False)
              .size()
              .reset_index(name="count")
              .sort_values("count", ascending=False)
              .reset_index(drop=True))

    counts.to_csv(OUTPUT_COUNTS_CSV, index=False)

    # 校验：频次和应等于帧总数
    total_frames = len(all_frames)
    sum_counts = int(counts["count"].sum())
    ok = "OK" if sum_counts == total_frames else f"MISMATCH (sum={sum_counts} vs frames={total_frames})"

    # 打印摘要与 Top-N
    print(f"[Info] Total frames (union): {total_frames}")
    print(f"[Info] Unique combos: {len(counts)}")
    print(f"[Check] Sum of combo counts == total frames?  {ok}\n")

    show_n = min(TOP_SHOW, len(counts))
    print(f"Top {show_n} combos:")
    for i in range(show_n):
        r = counts.iloc[i]
        print(f"{i+1:>2}. Car={r['Car_model']}, Pedestrian={r['Pedestrian_model']}, "
              f"Cyclist={r['Cyclist_model']}  |  count={int(r['count'])}")

    print(f"\n[Done] Per-frame combos  -> {OUTPUT_PER_FRAME_CSV}")
    print(f"[Done] Combo counts      -> {OUTPUT_COUNTS_CSV}")

if __name__ == "__main__":
    main()
