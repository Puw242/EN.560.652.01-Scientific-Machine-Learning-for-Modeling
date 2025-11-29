#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset Version 2:
- 从 9 个 KITTI detailed_tpfp JSONL 文件 (Car/Ped/Cyc × d0/d1/d2) 中，
  对指定的 tag(12/13/14) 对应的 frame_id 抽 60 个 case。
- 对于每个 frame：
    1) 保留该 frame 的所有 detection box（不再做 is_tp / ignored / iou_thr 过滤）；
    2) 同时对每个 class_name（Car / Pedestrian / Cyclist）标记一个 score 最高的 box 作为 best box。

输出：
- 每个 frame 一个 JSON 文件，包含:
    - frame_id, tag, tag_vector
    - detections: 所有 box 的完整列表
    - best: 每个类的 best box 概要信息
- 一个 index CSV，总结每个 case 的 best score（与 v1 有点类似）。
"""

# ================== INPUTS: 9 files (updated base dir) ==================
BASE = "/home/frank/Pu/sci_ML/kitti/Dataset_Prepare"

CAR_JSONL = [
    f"{BASE}/detailed_tpfp_Car_d0_iou0_70.jsonl",
    f"{BASE}/detailed_tpfp_Car_d1_iou0_70.jsonl",
    f"{BASE}/detailed_tpfp_Car_d2_iou0_70.jsonl",
]
PED_JSONL = [
    f"{BASE}/detailed_tpfp_Pedestrian_d0_iou0_50.jsonl",
    f"{BASE}/detailed_tpfp_Pedestrian_d1_iou0_50.jsonl",
    f"{BASE}/detailed_tpfp_Pedestrian_d2_iou0_50.jsonl",
]
CYC_JSONL = [
    f"{BASE}/detailed_tpfp_Cyclist_d0_iou0_50.jsonl",
    f"{BASE}/detailed_tpfp_Cyclist_d1_iou0_50.jsonl",
    f"{BASE}/detailed_tpfp_Cyclist_d2_iou0_50.jsonl",
]

ALL_JSONL = CAR_JSONL + PED_JSONL + CYC_JSONL

# ================== OUTPUTS (version 2) ==================
OUT_DIR = f"{BASE}/selected_cases_60_full_v2_all_boxes"
OUT_INDEX_CSV = f"{OUT_DIR}/selected_cases_60_full_v2_all_boxes_index.csv"

TOTAL_CASES = 60

# ======= Tag config: mapping and provided frame lists (string IDs) =======
# tag -> vector [Car, Ped, Cyc]; 0=None, 1=parta2, 2=pointrcnn, 3=ted
TAG_TO_VECTOR = {
    "12": [1, 2, 0],  # Car=parta2,    Ped=pointrcnn, Cyc=None
    "13": [2, 0, 2],  # Car=pointrcnn, Ped=None,      Cyc=pointrcnn
    "14": [3, 1, 0],  # Car=ted,       Ped=parta2,    Cyc=None
}

TAG12_FRAMES = ["1014","1356","1366","1499","1744","202","2122","2181","2275","2316",
                "2433","2656","2695","2830","2839","2913","2928","2983","3211","3273"]

TAG13_FRAMES = ["1105","122","123","1513","1522","1704","1878","2149","2179","2266",
                "2369","2512","2574","294","3125","3448","3654","522","609","647"]

TAG14_FRAMES = ["1234","1344","1562","1587","1664","1889","2015","2136","216","2171",
                "2523","2554","2742","299","3200","3386","3406","3475","3663","595"]

# ======================================================
import os, sys, json
import pandas as pd
from typing import List, Dict, Tuple

# 这些 key 按你原来的脚本做一个 sanity check
REQ = {"frame_id","class_name","iou_thr","iou","is_tp","ignored","score","det_idx"}

def _read_all_jsonl_with_diff(paths: List[str]) -> pd.DataFrame:
    """
    读取多个 jsonl 文件，把所有行合并到一个 DataFrame。
    为每一行增加:
      - '_difficulty': 从文件名解析出的 d0/d1/d2
      - '_src_file':   源文件名（可选，debug 用）
    不做任何过滤（is_tp / ignored / iou_thr），完全保留。
    """
    all_rows = []
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"[Error] File not found: {p}")
        # 解析 difficulty
        base = os.path.basename(p)
        diff = "d?"
        for dx in ["d0","d1","d2"]:
            if f"_{dx}_" in base or base.endswith(f"_{dx}.jsonl"):
                diff = dx
                break

        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError as e:
                    sys.exit(f"[Error] JSON decode failed at {p}:{ln}: {e}")

                missing = REQ - obj.keys()
                if missing:
                    sys.exit(
                        f"[Error] Missing keys {sorted(missing)} in {p}:{ln}, "
                        f"need={sorted(REQ)}"
                    )

                obj["_difficulty"] = diff
                obj["_src_file"] = base
                all_rows.append(obj)

        print(f"[Info] Loaded {p}")

    if not all_rows:
        return pd.DataFrame(columns=list(REQ) + ["_difficulty","_src_file"])

    df = pd.DataFrame(all_rows)

    # 类型统一
    df["frame_id"] = df["frame_id"].astype(str)
    for c in ["score", "iou", "iou_thr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_tp"] = df["is_tp"].astype(bool)
    df["ignored"] = df["ignored"].astype(bool)

    return df

def _ensure_outdir(path: str):
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    _ensure_outdir(OUT_DIR)
    _ensure_outdir(OUT_INDEX_CSV)

    # 1) 读入所有 jsonl（Car/Ped/Cyc × d0/d1/d2），完全保留所有 box
    all_df = _read_all_jsonl_with_diff(ALL_JSONL)

    if all_df.empty:
        sys.exit("[Error] No rows loaded from JSONL files.")

    # 所有出现过的 frame_id
    frames_union = set(all_df["frame_id"].unique())

    # 2) 预先计算每个 (frame_id, class_name) 的 best box（按 score 最大）
    best_map: Dict[Tuple[str, str], Dict] = {}
    for (fid, cls), grp in all_df.groupby(["frame_id","class_name"]):
        # 如果 score 里有 NaN，就先丢掉 NaN 再取 idxmax
        grp_valid = grp.dropna(subset=["score"])
        if grp_valid.empty:
            continue
        best_row = grp_valid.loc[grp_valid["score"].idxmax()]
        best_map[(str(fid), str(cls))] = best_row.to_dict()

    # 3) 按 tag 12 -> 13 -> 14 组装需要的 frame 列表
    ordered_targets = []
    used = set()

    def add_list(tag: str, ids: List[str]):
        for fid in ids:
            fid = str(fid)
            if fid in frames_union and fid not in used:
                ordered_targets.append((tag, fid))
                used.add(fid)

    add_list("12", TAG12_FRAMES)
    add_list("13", TAG13_FRAMES)
    add_list("14", TAG14_FRAMES)

    if len(ordered_targets) > TOTAL_CASES:
        ordered_targets = ordered_targets[:TOTAL_CASES]

    print(f"[Info] Prepared target frames: {len(ordered_targets)} (need {TOTAL_CASES})")

    # 4) 对每个 target frame 写 JSON + 记录 index
    index_rows = []
    for idx, (tag, fid) in enumerate(ordered_targets, start=1):
        frame_df = all_df[all_df["frame_id"] == fid].copy()

        if frame_df.empty:
            # 这个 frame 在 all_df 里没找到，写一个空的占位
            payload = {
                "frame_id": fid,
                "tag": tag,
                "tag_vector": TAG_TO_VECTOR[tag],
                "detections": [],
                "best": {
                    "Car": {"present": False},
                    "Pedestrian": {"present": False},
                    "Cyclist": {"present": False},
                },
            }
            car_score = ped_score = cyc_score = None
            car_present = ped_present = cyc_present = False
        else:
            # 排序一下：class_name, score(desc)
            frame_df = frame_df.sort_values(
                by=["class_name","score"], ascending=[True, False]
            )

            # 所有 box 全量保存
            detections = frame_df.to_dict(orient="records")

            best_info = {}
            best_scores = {}

            for cls in ["Car", "Pedestrian", "Cyclist"]:
                key = (fid, cls)
                if key in best_map:
                    b = best_map[key]
                    best_info[cls] = {
                        "present": True,
                        "score": None if pd.isna(b["score"]) else float(b["score"]),
                        "iou": None if pd.isna(b["iou"]) else float(b["iou"]),
                        "det_idx": None if pd.isna(b["det_idx"]) else int(b["det_idx"]),
                        "difficulty": str(b.get("_difficulty", "")),
                        "metric": b.get("metric", None),
                        "iou_thr": None if pd.isna(b["iou_thr"]) else float(b["iou_thr"]),
                        "is_tp": bool(b.get("is_tp", False)),
                        "ignored": bool(b.get("ignored", False)),
                    }
                    best_scores[cls] = best_info[cls]["score"]
                else:
                    best_info[cls] = {"present": False}
                    best_scores[cls] = None

            payload = {
                "frame_id": fid,
                "tag": tag,
                "tag_vector": TAG_TO_VECTOR[tag],
                "detections": detections,   # <<< 全部 box 在这里
                "best": best_info,          # <<< 每类一个 best box 概要
            }

            car_score = best_scores["Car"]
            ped_score = best_scores["Pedestrian"]
            cyc_score = best_scores["Cyclist"]
            car_present = best_info["Car"]["present"]
            ped_present = best_info["Pedestrian"]["present"]
            cyc_present = best_info["Cyclist"]["present"]

        # 写 per-case json
        _ensure_outdir(OUT_DIR)
        out_path = os.path.join(OUT_DIR, f"case_{idx:02d}_frame_{fid}_tag{tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # index 行：和 v1 类似，方便你快速看 60 个 frame 的整体情况
        index_rows.append({
            "case_id": idx,
            "frame_id": fid,
            "tag": tag,
            "tag_vector": str(TAG_TO_VECTOR[tag]),
            "Car_present": car_present,
            "Ped_present": ped_present,
            "Cyc_present": cyc_present,
            "Car_score": car_score,
            "Ped_score": ped_score,
            "Cyc_score": cyc_score,
        })

    # 5) 写 index CSV
    pd.DataFrame(index_rows).to_csv(OUT_INDEX_CSV, index=False)

    print(f"[Done] V2 (ALL BOXES) Wrote {len(index_rows)} case files -> {OUT_DIR}")
    print(f"[Done] V2 (ALL BOXES) Index CSV -> {OUT_INDEX_CSV}")

if __name__ == "__main__":
    main()
