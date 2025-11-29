#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析 tag / tag_vector 与 Car_score / Ped_score / Cyc_score 之间的相关性。

CSV 路径：
    /home/frank/Pu/sci_ML/kitti/Dataset_Prepare/selected_cases_60/selected_cases_60_index.csv

运行：
    python /home/frank/Pu/sci_ML/Tools/analyze_tag_score_corr.py
"""

import ast
import numpy as np
import pandas as pd

CSV_PATH = "/home/frank/Pu/sci_ML/kitti/Dataset_Prepare/selected_cases_60/selected_cases_60_index.csv"


def parse_tag_vector(s):
    """把字符串形式的 '[1, 2, 0]' 解析成 (1, 2, 0)。"""
    if pd.isna(s):
        return np.nan, np.nan, np.nan
    try:
        v = ast.literal_eval(s)
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            return np.nan, np.nan, np.nan
        return v[0], v[1], v[2]
    except Exception:
        return np.nan, np.nan, np.nan


def main():
    # 1. 读取 CSV
    df = pd.read_csv(CSV_PATH)

    # 2. 类型转换
    df["tag"] = pd.to_numeric(df["tag"], errors="coerce")

    for col in ["Car_score", "Ped_score", "Cyc_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. 解析 tag_vector -> tag_car, tag_ped, tag_cyc
    tag_car_list = []
    tag_ped_list = []
    tag_cyc_list = []

    for s in df["tag_vector"]:
        c, p, y = parse_tag_vector(s)
        tag_car_list.append(c)
        tag_ped_list.append(p)
        tag_cyc_list.append(y)

    df["tag_car"] = tag_car_list   # 对应 Car 的 expert id
    df["tag_ped"] = tag_ped_list   # 对应 Ped 的 expert id
    df["tag_cyc"] = tag_cyc_list   # 对应 Cyc 的 expert id

    # 4. 看一下前几行
    print("=== Head of Data ===")
    print(df.head(), "\n")

    # 5. 相关性分析 1：tag 与三个 score 的 Pearson 相关
    print("=== Pearson Correlation: tag vs scores ===")
    corr_tag = df[["tag", "Car_score", "Ped_score", "Cyc_score"]].corr()
    print(corr_tag, "\n")

    # 6. 相关性分析 2：tag_vector 分量 与 score 的 Pearson 相关
    print("=== Pearson Correlation: tag_vector components vs scores ===")
    cols_for_corr = [
        "tag_car",
        "tag_ped",
        "tag_cyc",
        "Car_score",
        "Ped_score",
        "Cyc_score",
    ]
    corr_tag_vec = df[cols_for_corr].corr()
    print(corr_tag_vec, "\n")

    # 7. 按 tag 分组的平均分数：看 tag=12/13/14 对应的平均 Car/Ped/Cyc score
    print("=== Mean scores grouped by tag ===")
    mean_by_tag = df.groupby("tag")[["Car_score", "Ped_score", "Cyc_score"]].mean()
    print(mean_by_tag, "\n")

    # 8. 可选：按 tag_vector 的 expert id 分组看平均分
    print("=== Mean Car_score grouped by tag_car (Car expert id) ===")
    print(df.groupby("tag_car")[["Car_score"]].mean(), "\n")

    print("=== Mean Ped_score grouped by tag_ped (Ped expert id) ===")
    print(df.groupby("tag_ped")[["Ped_score"]].mean(), "\n")

    print("=== Mean Cyc_score grouped by tag_cyc (Cyc expert id) ===")
    print(df.groupby("tag_cyc")[["Cyc_score"]].mean(), "\n")


if __name__ == "__main__":
    main()
