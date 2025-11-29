#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select top-K nuScenes matching cases per model.

运行方式：
    python select_top_cases.py

你可以只改下面前四个常量即可（输入/输出路径 & K 值）。
"""

# ========= 路径与参数（放在最前面，按需修改） =========
INPUT_CSV      = "/home/frank/Pu/sci_ML/nuscenses/best_model_per_token.csv"  # 输入CSV
OUTPUT_MERGED  = "/home/frank/Pu/sci_ML/nuscenses/top60.csv"                 # 合并后的输出CSV
OUTPUT_DIR     = "/home/frank/Pu/sci_ML/nuscenses/top_by_model"              # 每个模型单独导出的目录
TOP_K_PER_MODEL = 20                                                          # 每个模型取前K条
SPLIT_PER_MODEL = True                                                        # 是否额外导出每模型文件
# ====================================================

import os
import sys
import pandas as pd

REQUIRED_COLS = {"sample_token", "best_model", "best_nd_score"}

def ensure_parent_dir(path: str):
    """确保输出文件/目录的父目录存在。"""
    parent = path if os.path.isdir(path) else os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        sys.exit(f"[Error] Missing columns: {sorted(missing)} in {csv_path}. "
                 f"Required: {sorted(REQUIRED_COLS)}")
    # 清洗：去空、类型规范、同一(model, token)保留分数最高的
    df = df.dropna(subset=list(REQUIRED_COLS)).copy()
    df["best_model"] = df["best_model"].astype(str)
    df["best_nd_score"] = df["best_nd_score"].astype(float)
    df = df.sort_values(["best_model", "sample_token", "best_nd_score"],
                        ascending=[True, True, False])
    df = df.drop_duplicates(subset=["best_model", "sample_token"], keep="first")
    return df

def select_top_per_model(df: pd.DataFrame, k: int) -> pd.DataFrame:
    tops = []
    summary = []
    for model, g in df.groupby("best_model", sort=True):
        g_sorted = g.sort_values("best_nd_score", ascending=False).head(k).copy()
        g_sorted.insert(0, "rank_in_model", range(1, len(g_sorted) + 1))
        tops.append(g_sorted)
        summary.append((model, len(g_sorted)))
    if not tops:
        sys.exit("[Error] No groups found. Check 'best_model' values.")
    out = pd.concat(tops, ignore_index=True)
    out = out.sort_values(["best_model", "best_nd_score"],
                          ascending=[True, False])
    # 分数格式化为6位小数，便于比对
    out["best_nd_score"] = out["best_nd_score"].map(lambda x: float(f"{x:.6f}"))
    return out, summary

def write_per_model_files(df: pd.DataFrame, out_dir: str):
    ensure_parent_dir(out_dir)
    for model, g in df.groupby("best_model", sort=True):
        outm = os.path.join(out_dir, f"top_{model}.csv".replace("/", "_"))
        g.to_csv(outm, index=False)

def main():
    print(f"[Info] Input : {INPUT_CSV}")
    print(f"[Info] Output(merged): {OUTPUT_MERGED}")
    print(f"[Info] Output(per-model dir): {OUTPUT_DIR} (enabled={SPLIT_PER_MODEL})")
    print(f"[Info] Top-K per model: {TOP_K_PER_MODEL}")

    if not os.path.exists(INPUT_CSV):
        sys.exit(f"[Error] Input CSV not found: {INPUT_CSV}")

    ensure_parent_dir(OUTPUT_MERGED)
    if SPLIT_PER_MODEL:
        ensure_parent_dir(OUTPUT_DIR)

    df = load_and_clean(INPUT_CSV)
    out, summary = select_top_per_model(df, TOP_K_PER_MODEL)

    # 统一列顺序
    cols = ["rank_in_model", "best_model", "sample_token", "best_nd_score"]
    out = out[cols]
    out.to_csv(OUTPUT_MERGED, index=False)

    if SPLIT_PER_MODEL:
        write_per_model_files(out, OUTPUT_DIR)

    total = len(out)
    detail = ", ".join([f"{m}:{c}" for m, c in summary])
    print(f"[Done] Wrote merged top cases: {total} rows -> {OUTPUT_MERGED}")
    print(f"[Info] Per-model counts (requested top {TOP_K_PER_MODEL}): {detail}")
    if SPLIT_PER_MODEL:
        print(f"[Done] Per-model files saved under: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
