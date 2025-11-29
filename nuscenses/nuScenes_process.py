# proces#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import defaultdict

INPUT_FILE = "/home/frank/Pu/sci_ML/nuscenses/best_model_per_token.csv"   # <<< 修改为你的真实路径

def main():
    count = defaultdict(int)
    score_sum = defaultdict(float)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["best_model"]
            score = float(row["best_nd_score"])
            count[model] += 1
            score_sum[model] += score

    print("Model\tCount\tMean Score")
    for model in sorted(count.keys()):
        c = count[model]
        mean = score_sum[model] / c if c else 0.0
        print(f"{model}\t{c}\t{mean:.6f}")

if __name__ == "__main__":
    main()
