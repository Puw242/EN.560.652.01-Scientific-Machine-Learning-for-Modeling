#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge three model top-case CSVs into one file.

运行方式：
    python merge_top_cases.py
"""

# ========= 文件路径（按需修改） =========
INPUT_FILES = [
    "/home/frank/Pu/sci_ML/nuscenses/top_by_model/top_transfusion.csv",
    "/home/frank/Pu/sci_ML/nuscenses/top_by_model/top_BEVfusion.csv",
    "/home/frank/Pu/sci_ML/nuscenses/top_by_model/top_BEVDAL.csv"
]

OUTPUT_FILE = "/home/frank/Pu/sci_ML/nuscenses/top60_merged.csv"
# ======================================

import os
import sys
import pandas as pd

def main():
    # 检查输入文件
    missing = [f for f in INPUT_FILES if not os.path.exists(f)]
    if missing:
        sys.exit(f"[Error] Missing file(s):\n" + "\n".join(missing))

    dfs = []
    for path in INPUT_FILES:
        print(f"[Info] Reading: {path}")
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        dfs.append(df)

    # 合并
    merged = pd.concat(dfs, ignore_index=True)
    # 排序（可选）：按模型名和 best_nd_score 降序
    if "best_model" in merged.columns and "best_nd_score" in merged.columns:
        merged = merged.sort_values(["best_model", "best_nd_score"], ascending=[True, False])

    # 输出目录检查
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"[Done] Wrote merged file with {len(merged)} rows -> {OUTPUT_FILE}")

    # 打印每个文件的行数摘要
    for path, df in zip(INPUT_FILES, dfs):
        print(f"[Info] {os.path.basename(path)}: {len(df)} rows")

if __name__ == "__main__":
    main()
