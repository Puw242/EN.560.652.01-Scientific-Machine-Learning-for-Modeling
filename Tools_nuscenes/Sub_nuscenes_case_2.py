# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 从 merged_top_samples.csv 中选出 detection_scores 最高的 10 个样本。

# 依据：
#   对每一行的 detection_scores（一个 list）取 max，
#   然后按 max_score 排序，保留前 10。

# 输入：
#   /home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples.csv

# 输出：
#   /home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10_by_score.csv
# """

# import os
# import ast
# import pandas as pd

# BASE_DIR = "/home/frank/Pu/sci_ML/nuscenses/top_by_model"
# IN_CSV  = os.path.join(BASE_DIR, "merged_top_samples.csv")
# OUT_CSV = os.path.join(BASE_DIR, "merged_top_samples_top10_by_score.csv")

# def parse_detection_scores(s):
#     """
#     把字符串形式的 "[0.1, 0.2, ...]" 解析成 list[float]，
#     返回 list；如果解析失败或为空，则返回空 list。
#     """
#     if pd.isna(s):
#         return []
#     s = str(s).strip()
#     if not s:
#         return []
#     try:
#         scores = ast.literal_eval(s)
#         # 有时候是单个 float，也兼容一下
#         if isinstance(scores, (int, float)):
#             return [float(scores)]
#         if isinstance(scores, (list, tuple)):
#             return [float(x) for x in scores]
#         # 其他情况就当空
#         return []
#     except Exception:
#         return []

# def main():
#     if not os.path.exists(IN_CSV):
#         raise FileNotFoundError(f"[Error] Input CSV not found: {IN_CSV}")

#     df = pd.read_csv(IN_CSV)
#     if "detection_scores" not in df.columns:
#         raise ValueError("[Error] CSV 中没有 detection_scores 列")

#     # 解析 detection_scores 并计算每一行的最大分数
#     max_scores = []
#     for i, s in enumerate(df["detection_scores"]):
#         scores = parse_detection_scores(s)
#         if scores:
#             max_scores.append(max(scores))
#         else:
#             # 如果没有分数，就设为 0（也可以设为 -inf）
#             max_scores.append(0.0)

#     df["max_det_score"] = max_scores

#     # 按 max_det_score 从大到小排序，取前 10
#     df_top10 = df.sort_values("max_det_score", ascending=False).head(10).reset_index(drop=True)

#     # 保存结果
#     df_top10.to_csv(OUT_CSV, index=False)
#     print(f"[Info] 原始样本数 = {len(df)}")
#     print(f"[Info] 按 detection_scores 最大值选出前 10 个样本")
#     print(f"[Done] 已保存到: {OUT_CSV}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对每个 sample_token，只保留 detection_scores 最高的 10 个检测，
但保留所有样本行（60 行）。

输入：
  /home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples.csv

输出：
  /home/frank/Pu/sci_ML/nuscenses/top_by_model/merged_top_samples_top10det.csv

输出列：
  sample_token,num_detections,detection_names,detection_scores,
  boxes_lidar,pred_labels,rank_in_model,best_model,best_nd_score,max_det_score
"""

import os
import ast
import pandas as pd

BASE_DIR = "/home/frank/Pu/sci_ML/nuscenses/top_by_model"
IN_CSV  = os.path.join(BASE_DIR, "merged_top_samples.csv")
OUT_CSV = os.path.join(BASE_DIR, "merged_top_samples_top10det.csv")

TOP_K = 10  # 每个 sample 内保留的检测数

def parse_list(s):
    """把字符串形式的 list 解析成 Python list."""
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    return ast.literal_eval(s)

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    print(f"[Info] 原始样本数: {len(df)}")

    new_rows = []

    for idx, row in df.iterrows():
        det_names  = parse_list(row["detection_names"])
        det_scores = parse_list(row["detection_scores"])
        boxes      = parse_list(row["boxes_lidar"])
        labels     = parse_list(row["pred_labels"])

        n = len(det_scores)
        if not (len(det_names) == len(boxes) == len(labels) == n):
            print(f"[Warn] 第 {idx} 行检测长度不一致，跳过对齐检查，尝试按 scores 长度裁剪")
            det_names  = det_names[:n]
            boxes      = boxes[:n]
            labels     = labels[:n]

        if n == 0:
            max_score = 0.0
            top_names, top_scores, top_boxes, top_labels = [], [], [], []
        else:
            # 根据 score 排序（降序），取前 TOP_K
            order = sorted(range(n), key=lambda i: det_scores[i], reverse=True)
            top_idx = order[:TOP_K]

            top_names  = [det_names[i]  for i in top_idx]
            top_scores = [det_scores[i] for i in top_idx]
            top_boxes  = [boxes[i]      for i in top_idx]
            top_labels = [labels[i]     for i in top_idx]

            max_score = max(top_scores)

        new_row = row.copy()
        new_row["detection_names"]  = str(top_names)
        new_row["detection_scores"] = str(top_scores)
        new_row["boxes_lidar"]      = str(top_boxes)
        new_row["pred_labels"]      = str(top_labels)
        new_row["num_detections"]   = len(top_scores)
        new_row["max_det_score"]    = max_score

        new_rows.append(new_row)

    df_out = pd.DataFrame(new_rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"[Done] 每个 sample 仅保留 top-{TOP_K} detections")
    print(f"[Done] 结果已保存到: {OUT_CSV}")

if __name__ == "__main__":
    main()

