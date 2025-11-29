#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 nuScenes 大预测文件中抽取 top-by-model 小数据集对应的样本。

输入：
    小 CSV：
        /home/frank/Pu/sci_ML/nuscenses/top_by_model/top_BEVDAL.csv
        /home/frank/Pu/sci_ML/nuscenses/top_by_model/top_BEVfusion.csv
        /home/frank/Pu/sci_ML/nuscenses/top_by_model/top_transfusion.csv

        列结构：
        rank_in_model,best_model,sample_token,best_nd_score

    大 CSV：
        /home/frank/Pu/sci_ML/nuscenses/merged_predictions.csv

        至少包含列：
        sample_token,num_detections,detection_names,detection_scores,boxes_lidar,pred_labels

输出：
    1) 只包含 top 样本的预测结果：
       /home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples.csv

    2) 可选：把 best_model / best_nd_score 合并进来，一并保存在同一个文件里。
"""

import os
import pandas as pd

# ======== 路径配置 ========
BASE = "/home/frank/Pu/sci_ML/nuscenses"
TOP_DIR = os.path.join(BASE, "top_by_model")

TOP_FILES = [
    os.path.join(TOP_DIR, "top_BEVDAL.csv"),
    os.path.join(TOP_DIR, "top_BEVfusion.csv"),
    os.path.join(TOP_DIR, "top_transfusion.csv"),
]

BIG_CSV = os.path.join(BASE, "merged_predictions.csv")

OUT_CSV = os.path.join(TOP_DIR, "merged_top_samples.csv")


def load_top_tokens(top_files):
    """
    读取三个 top_by_model CSV，返回：
      - df_top_all: 合并后的 DataFrame
      - token_set:  所有 sample_token 的集合（去重）
    """
    dfs = []
    for p in top_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[Error] Not found: {p}")
        df = pd.read_csv(p)
        # 防止列名问题，这里简单检查一下
        required_cols = {"rank_in_model", "best_model", "sample_token", "best_nd_score"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"[Error] {p} 缺少列: {required_cols - set(df.columns)}")
        dfs.append(df)

    df_top_all = pd.concat(dfs, ignore_index=True)
    # 去重：同一个 sample_token 可能在不同模型的 top 里
    # 如果你想保留多个条目（每个模型一条），直接用 df_top_all 不去重也可以
    token_set = set(df_top_all["sample_token"].astype(str).tolist())
    print(f"[Info] 合并 top_by_model，共有 {len(df_top_all)} 条（含重复），"
          f"unique sample_token = {len(token_set)}")
    return df_top_all, token_set


def filter_big_csv_by_tokens(big_csv, token_set, df_top_all):
    """
    在大文件 merged_predictions.csv 中筛选出 sample_token 落在 token_set 的行。

    为避免一次性读入太大，用 chunksize 分块读取。
    同时把 df_top_all 的信息 merge 进去（按 sample_token 对齐）。
    """
    if not os.path.exists(big_csv):
        raise FileNotFoundError(f"[Error] Not found: {big_csv}")

    chunks = []
    # 根据你的机器内存情况调整 chunksize
    for i, chunk in enumerate(pd.read_csv(big_csv, chunksize=1000)):
        # 确保 sample_token 是字符串形式
        chunk["sample_token"] = chunk["sample_token"].astype(str)
        sub = chunk[chunk["sample_token"].isin(token_set)].copy()
        if not sub.empty:
            chunks.append(sub)
        print(f"[Info] 处理分块 {i}, 当前选中 {len(sub)} 行")

    if not chunks:
        print("[Warn] 没有在 merged_predictions.csv 里找到任何匹配的 sample_token")
        return None

    df_filtered = pd.concat(chunks, ignore_index=True)
    print(f"[Info] 过滤后总共得到 {len(df_filtered)} 行预测结果")

    # 把 best_model / best_nd_score 等信息 merge 进来
    # 注意：df_top_all 里同一个 sample_token 可能出现多次（不同 best_model），
    # 这里的策略是：按 sample_token merge，可能会产生多行（笛卡尔积），
    # 如果你只想保留 "best_nd_score 最大的那一个"，下面做个简单处理。
    df_top_all = df_top_all.copy()
    df_top_all["sample_token"] = df_top_all["sample_token"].astype(str)

    # 对于同一个 sample_token，保留 nd_score 最大的那一条
    df_top_best = (
        df_top_all.sort_values(["sample_token", "best_nd_score"], ascending=[True, False])
                  .drop_duplicates(subset=["sample_token"], keep="first")
    )

    print(f"[Info] df_top_best 保留每个 sample_token 一条，行数 = {len(df_top_best)}")

    # merge 回 filtered prediction
    df_merged = df_filtered.merge(df_top_best, on="sample_token", how="left")

    return df_merged


def main():
    # 1. 读三个小 CSV，拿到 token_set
    df_top_all, token_set = load_top_tokens(TOP_FILES)

    # 2. 在大文件中筛选
    df_merged = filter_big_csv_by_tokens(BIG_CSV, token_set, df_top_all)
    if df_merged is None:
        return

    # 3. 存结果
    os.makedirs(TOP_DIR, exist_ok=True)
    df_merged.to_csv(OUT_CSV, index=False)
    print(f"[Done] 已保存 subset 到: {OUT_CSV}")
    print("[Done] 列包含 sample_token 原始预测 + best_model/best_nd_score 等 top 信息")


if __name__ == "__main__":
    main()
