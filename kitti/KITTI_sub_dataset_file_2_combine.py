#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find frames matching specified (Car, Pedestrian, Cyclist) combos and tag each frame by category.

改动点：
- 仍按每条记录自带 iou_thr 过滤（Car=0.7 通常在数据里；Ped/Cyc=0.5）
- 合并三类后对缺失类填 'none'
- 根据你给的三种组合：
    12. (car=parta2,    ped=pointrcnn, cyclist=none)
    13. (car=pointrcnn, ped=none,      cyclist=pointrcnn)
    14. (car=ted,       ped=parta2,    cyclist=none)
  找出所有匹配帧，并在输出中标注属于 {12,13,14} 哪一类

运行：
    python /home/frank/Pu/sci_ML/kitti/KITTI_find_combo_frames.py
"""

# ================== 输入/输出路径（按需修改） ==================
CAR_JSONL = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Car.jsonl"
PED_JSONL = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Pedestrian.jsonl"
CYC_JSONL = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Cyclist.jsonl"

# 合并后的每帧明细（含 none）也会导出，便于复用
OUTPUT_PER_FRAME_CSV   = "/home/frank/Pu/sci_ML/kitti/kitti_triplet_combos_per_frame.csv"
# 只包含命中三种组合的帧（标注类别 12/13/14）
OUTPUT_MATCHED_FRAMES  = "/home/frank/Pu/sci_ML/kitti/kitti_frames_matched_12_13_14.csv"
# 统计摘要（每一类的计数）
OUTPUT_MATCHED_COUNTS  = "/home/frank/Pu/sci_ML/kitti/kitti_combo_12_13_14_counts.csv"
TOP_SHOW = 20
# ============================================================

import os
import sys
import json
import pandas as pd

REQUIRED_KEYS = {
    "frame_id", "class_name", "iou_thr", "iou",
    "is_tp", "ignored", "score", "_selected_model"
}

# 目标组合（小写），以及对应的类别标签
TARGET_COMBOS = {
    # combo_tuple (car, ped, cyc) : tag
    ("parta2",    "pointrcnn", "none"): "12",
    ("pointrcnn", "none",      "pointrcnn"): "13",
    ("ted",       "parta2",    "none"): "14",
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
    all_frames = set(df["frame_id"].astype(str))

    df = df[df["class_name"] == expect_class].copy()
    if df.empty:
        # 该类没有记录，返回空DF和帧集合
        return pd.DataFrame(columns=["frame_id", f"{expect_class}_model", f"{expect_class}_score"]), all_frames

    # 类型 & 清洗
    df["frame_id"] = df["frame_id"].astype(str)
    df["iou_thr"] = pd.to_numeric(df["iou_thr"], errors="coerce")
    df["iou"] = pd.to_numeric(df["iou"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["_selected_model"] = df["_selected_model"].astype(str).str.strip().str.lower()
    df["is_tp"] = df["is_tp"].astype(bool)
    df["ignored"] = df["ignored"].astype(bool)

    # 过滤（逐记录）：iou >= iou_thr 且 is_tp 且 not ignored
    before = len(df)
    df = df[(df["iou"] >= df["iou_thr"]) & (df["is_tp"]) & (~df["ignored"])].copy()
    after = len(df)
    print(f"[Info] {expect_class}: rows before={before}, after_filter={after}, file={os.path.basename(path)}")

    if df.empty:
        return pd.DataFrame(columns=["frame_id", f"{expect_class}_model", f"{expect_class}_score"]), all_frames

    # 每帧取最高分
    df = df.sort_values(["frame_id", "score"], ascending=[True, False]).drop_duplicates(subset=["frame_id"], keep="first")

    # 保留字段并规范命名
    keep = df[["frame_id", "_selected_model", "score"]].copy()
    keep.rename(columns={
        "_selected_model": f"{expect_class}_model",
        "score": f"{expect_class}_score"
    }, inplace=True)
    return keep, all_frames

def main():
    # 输出目录
    for p in [OUTPUT_PER_FRAME_CSV, OUTPUT_MATCHED_FRAMES, OUTPUT_MATCHED_COUNTS]:
        out_dir = os.path.dirname(p)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    # 读取三类
    car_df, frames_car = load_df(CAR_JSONL, "Car")
    ped_df, frames_ped = load_df(PED_JSONL, "Pedestrian")
    cyc_df, frames_cyc = load_df(CYC_JSONL, "Cyclist")

    # 所有出现过的帧（并集）
    all_frames = sorted(frames_car | frames_ped | frames_cyc)
    if not all_frames:
        sys.exit("[Error] No frame_id found in any input.")

    # 骨架 + 外连接
    base = pd.DataFrame({"frame_id": all_frames})
    merged = base.merge(car_df, on="frame_id", how="left") \
                 .merge(ped_df, on="frame_id", how="left") \
                 .merge(cyc_df, on="frame_id", how="left")

    # 缺失类填 'none'，模型名统一小写
    for col in ["Car_model", "Pedestrian_model", "Cyclist_model"]:
        if col not in merged.columns:
            merged[col] = "none"
        else:
            merged[col] = merged[col].fillna("none").astype(str).str.strip().str.lower()

    # 保存每帧明细（含 none）
    keep_cols = ["frame_id",
                 "Car_model", "Car_score",
                 "Pedestrian_model", "Pedestrian_score",
                 "Cyclist_model", "Cyclist_score"]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged[keep_cols].to_csv(OUTPUT_PER_FRAME_CSV, index=False)

    # 给每一帧打组合标签（12/13/14 或空）
    merged["combo_tuple"] = list(zip(merged["Car_model"], merged["Pedestrian_model"], merged["Cyclist_model"]))
    merged["combo_tag"] = merged["combo_tuple"].map(TARGET_COMBOS).fillna("")

    # 仅保留命中 12/13/14 的帧
    matched = merged[merged["combo_tag"] != ""].copy()

    # 输出命中帧明细
    out_cols = ["frame_id", "combo_tag", "Car_model", "Pedestrian_model", "Cyclist_model",
                "Car_score", "Pedestrian_score", "Cyclist_score"]
    matched[out_cols].to_csv(OUTPUT_MATCHED_FRAMES, index=False)

    # 统计每一类的计数
    counts = matched.groupby("combo_tag").size().reset_index(name="count").sort_values("combo_tag")
    counts.to_csv(OUTPUT_MATCHED_COUNTS, index=False)

    # 终端摘要
    print(f"[Info] Total frames (union): {len(all_frames)}")
    print(f"[Info] Matched frames total: {len(matched)}")
    if not counts.empty:
        print("\nCounts by category (12/13/14):")
        for _, row in counts.iterrows():
            print(f"  tag={row['combo_tag']}  count={int(row['count'])}")

    # 打印每类前若干帧例子
    for tag in ["12", "13", "14"]:
        sub = matched[matched["combo_tag"] == tag]
        if not sub.empty:
            show_n = min(TOP_SHOW, len(sub))
            print(f"\n[Examples] tag {tag}  (show {show_n}/{len(sub)}):")
            print(", ".join(sub["frame_id"].head(show_n).tolist()))

    print(f"\n[Done] Per-frame detail  -> {OUTPUT_PER_FRAME_CSV}")
    print(f"[Done] Matched frames    -> {OUTPUT_MATCHED_FRAMES}")
    print(f"[Done] Matched counts    -> {OUTPUT_MATCHED_COUNTS}")

if __name__ == "__main__":
    main()
