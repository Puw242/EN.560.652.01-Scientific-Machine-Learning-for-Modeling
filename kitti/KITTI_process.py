#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import defaultdict

# 固定 jsonl 文件路径
INPUT_FILE = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Pedestrian.jsonl"


# INPUT_FILE = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Pedestrian.jsonl"
# INPUT_FILE = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Car.jsonl"
# INPUT_FILE = "/home/frank/Pu/sci_ML/kitti/BEST_SELECTION_Cyclist.jsonl"


def main():

    count = defaultdict(int)
    iou_sum = defaultdict(float)

    total = 0
    counted = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except:
                continue

            total += 1

            model = obj.get("_selected_model", None)
            iou = obj.get("iou", None)
            thr = obj.get("iou_thr", None)

            if model is None or iou is None or thr is None:
                continue

            # 只有 iou >= iou_thr 才计数
            if float(iou) >= float(thr):
                count[model] += 1
                iou_sum[model] += float(iou)
                counted += 1

    print(f"\nTotal records: {total}, Counted (iou >= iou_thr): {counted}\n")
    print("Model\tCount\tMean IOU")

    for model in sorted(count.keys()):
        c = count[model]
        mean = iou_sum[model] / c if c > 0 else 0.0
        print(f"{model}\t{c}\t{mean:.6f}")

if __name__ == "__main__":
    main()
