## 这里是 subset version 1， 我们再写一个subset version 2 还是 同样的


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save 60 per-frame 'cases' (one file per frame) selected from 9 KITTI detailed_tpfp JSONL files
(Car/Ped/Cyc × d0/d1/d2). Frames are taken from three provided tag lists (12/13/14) in order
until reaching 60. Each output JSON includes the tag, tag_vector, and per-class best detection.

Run:
    python /home/frank/Pu/sci_ML/kitti/KITTI_save_60_tagged_cases.py
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

# ================== OUTPUTS (in same base dir) ==================
OUT_DIR = f"{BASE}/selected_cases_60"                # each case -> one JSON
OUT_INDEX_CSV = f"{OUT_DIR}/selected_cases_60_index.csv"

TOTAL_CASES = 60

# ======= Tag config: mapping and provided frame lists (string IDs) =======
# tag -> vector [Car, Ped, Cyc]; 0=None, 1=parta2, 2=pointrcnn, 3=ted
TAG_TO_VECTOR = {
    "12": [1, 2, 0],  # Car=parta2,    Ped=pointrcnn, Cyc=None
    "13": [2, 0, 2],  # Car=pointrcnn, Ped=None,      Cyc=pointrcnn
    "14": [3, 1, 0],  # Car=ted,       Ped=parta2,    Cyc=None
}

# You provided these frame_id lists (show 20 each). Append more if needed.
TAG12_FRAMES = ["1014","1356","1366","1499","1744","202","2122","2181","2275","2316",
                "2433","2656","2695","2830","2839","2913","2928","2983","3211","3273"]

TAG13_FRAMES = ["1105","122","123","1513","1522","1704","1878","2149","2179","2266",
                "2369","2512","2574","294","3125","3448","3654","522","609","647"]

TAG14_FRAMES = ["1234","1344","1562","1587","1664","1889","2015","2136","216","2171",
                "2523","2554","2742","299","3200","3386","3406","3475","3663","595"]

# ======================================================
import os, sys, json
import pandas as pd
from typing import List, Dict

REQ = {"frame_id","class_name","iou_thr","iou","is_tp","ignored","score","det_idx"}

def _read_jsonl(path: str):
    if not os.path.exists(path):
        sys.exit(f"[Error] File not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                sys.exit(f"[Error] JSON decode failed at {path}:{ln}: {e}")
            rows.append(obj)
    return rows

def _collect_best_across_diffs(paths: List[str], expect_class: str) -> pd.DataFrame:
    """
    Merge multiple difficulty jsonl for a class, filter valid detections, and keep the
    BEST per frame (highest score) across d0/d1/d2.
    Returns columns:
      frame_id, {Class}_present, {Class}_score, {Class}_iou, {Class}_det_idx, {Class}_difficulty
    """
    dfs = []
    for p in paths:
        rows = _read_jsonl(p)
        if not rows:
            continue
        bad = [i for i, r in enumerate(rows) if not REQ.issubset(r)]
        if bad:
            sys.exit(f"[Error] Missing keys in {p}, e.g. idx={bad[0]}; need={sorted(REQ)}")
        df = pd.DataFrame(rows)
        df = df[df["class_name"] == expect_class].copy()
        if df.empty:
            continue
        df["frame_id"] = df["frame_id"].astype(str)
        for c in ["iou_thr","iou","score"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["is_tp"] = df["is_tp"].astype(bool)
        df["ignored"] = df["ignored"].astype(bool)

        before = len(df)
        df = df[(df["is_tp"]) & (~df["ignored"]) & (df["iou"] >= df["iou_thr"])].copy()
        after = len(df)
        print(f"[Info] {expect_class} <- {os.path.basename(p)}: before={before}, after_valid={after}")

        # parse difficulty (d0/d1/d2) from filename
        diff = "d?"
        base = os.path.basename(p)
        for dx in ["d0","d1","d2"]:
            if f"_{dx}_" in base or base.endswith(f"_{dx}.jsonl"):
                diff = dx
                break
        df["_difficulty"] = diff
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["frame_id", f"{expect_class}_score",
                                     f"{expect_class}_iou", f"{expect_class}_det_idx",
                                     f"{expect_class}_difficulty", f"{expect_class}_present"])

    cat = pd.concat(dfs, ignore_index=True)
    if cat.empty:
        return pd.DataFrame(columns=["frame_id", f"{expect_class}_score",
                                     f"{expect_class}_iou", f"{expect_class}_det_idx",
                                     f"{expect_class}_difficulty", f"{expect_class}_present"])

    # keep one (best score) per frame across difficulties
    cat = cat.sort_values(["frame_id","score"], ascending=[True, False]) \
             .drop_duplicates(subset=["frame_id"], keep="first")

    out = (cat[["frame_id","score","iou","det_idx","_difficulty"]]
           .rename(columns={
               "score": f"{expect_class}_score",
               "iou": f"{expect_class}_iou",
               "det_idx": f"{expect_class}_det_idx",
               "_difficulty": f"{expect_class}_difficulty",
           })
           .assign(**{f"{expect_class}_present": True})
           )
    return out

def _ensure_outdir(path: str):
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    _ensure_outdir(OUT_DIR)
    _ensure_outdir(OUT_INDEX_CSV)

    # collect per-class best over d0/d1/d2
    car_best = _collect_best_across_diffs(CAR_JSONL, "Car")
    ped_best = _collect_best_across_diffs(PED_JSONL, "Pedestrian")
    cyc_best = _collect_best_across_diffs(CYC_JSONL, "Cyclist")

    # union of all frames that appear in any of the 9 jsonl
    frames_union = set()
    for lst in [CAR_JSONL, PED_JSONL, CYC_JSONL]:
        for p in lst:
            for r in _read_jsonl(p):
                if "frame_id" in r:
                    frames_union.add(str(r["frame_id"]))

    base = pd.DataFrame({"frame_id": sorted(frames_union)})

    # merge class-best tables
    merged = (base.merge(car_best, on="frame_id", how="left")
                   .merge(ped_best, on="frame_id", how="left")
                   .merge(cyc_best, on="frame_id", how="left"))

    # fill missing presence + numeric columns
    for cls in ["Car","Pedestrian","Cyclist"]:
        pcol = f"{cls}_present"
        if pcol not in merged:
            merged[pcol] = False
        else:
            merged[pcol] = merged[pcol].fillna(False).astype(bool)
        for c in [f"{cls}_score", f"{cls}_iou", f"{cls}_det_idx", f"{cls}_difficulty"]:
            if c not in merged:
                merged[c] = pd.NA

    # assemble target order: tag 12 -> tag 13 -> tag 14
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

    # trim to TOTAL_CASES
    if len(ordered_targets) > TOTAL_CASES:
        ordered_targets = ordered_targets[:TOTAL_CASES]

    print(f"[Info] Prepared target frames: {len(ordered_targets)} (need {TOTAL_CASES})")

    # write 60 JSON files + index CSV
    index_rows = []
    for idx, (tag, fid) in enumerate(ordered_targets, start=1):
        row = merged[merged["frame_id"] == fid]
        if row.empty:
            payload = {
                "frame_id": fid,
                "tag": tag,
                "tag_vector": TAG_TO_VECTOR[tag],
                "Car": {"present": False},
                "Pedestrian": {"present": False},
                "Cyclist": {"present": False},
            }
        else:
            r = row.iloc[0]
            payload = {
                "frame_id": fid,
                "tag": tag,
                "tag_vector": TAG_TO_VECTOR[tag],  # e.g. [1,2,0] / [2,0,2] / [3,1,0]
                "Car": {
                    "present": bool(r["Car_present"]),
                    "score": None if pd.isna(r["Car_score"]) else float(r["Car_score"]),
                    "iou": None if pd.isna(r["Car_iou"]) else float(r["Car_iou"]),
                    "det_idx": None if pd.isna(r["Car_det_idx"]) else int(r["Car_det_idx"]),
                    "difficulty": None if pd.isna(r["Car_difficulty"]) else str(r["Car_difficulty"]),
                },
                "Pedestrian": {
                    "present": bool(r["Pedestrian_present"]),
                    "score": None if pd.isna(r["Pedestrian_score"]) else float(r["Pedestrian_score"]),
                    "iou": None if pd.isna(r["Pedestrian_iou"]) else float(r["Pedestrian_iou"]),
                    "det_idx": None if pd.isna(r["Pedestrian_det_idx"]) else int(r["Pedestrian_det_idx"]),
                    "difficulty": None if pd.isna(r["Pedestrian_difficulty"]) else str(r["Pedestrian_difficulty"]),
                },
                "Cyclist": {
                    "present": bool(r["Cyclist_present"]),
                    "score": None if pd.isna(r["Cyclist_score"]) else float(r["Cyclist_score"]),
                    "iou": None if pd.isna(r["Cyclist_iou"]) else float(r["Cyclist_iou"]),
                    "det_idx": None if pd.isna(r["Cyclist_det_idx"]) else int(r["Cyclist_det_idx"]),
                    "difficulty": None if pd.isna(r["Cyclist_difficulty"]) else str(r["Cyclist_difficulty"]),
                },
            }

        # write per-case json
        _ensure_outdir(OUT_DIR)
        out_path = os.path.join(OUT_DIR, f"case_{idx:02d}_frame_{fid}_tag{tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # index row
        index_rows.append({
            "case_id": idx,
            "frame_id": fid,
            "tag": tag,
            "tag_vector": str(TAG_TO_VECTOR[tag]),
            "Car_present": payload["Car"]["present"],
            "Ped_present": payload["Pedestrian"]["present"],
            "Cyc_present": payload["Cyclist"]["present"],
            "Car_score": payload["Car"]["score"],
            "Ped_score": payload["Pedestrian"]["score"],
            "Cyc_score": payload["Cyclist"]["score"],
        })

    # write index csv
    pd.DataFrame(index_rows).to_csv(OUT_INDEX_CSV, index=False)

    print(f"[Done] Wrote {len(index_rows)} case files -> {OUT_DIR}")
    print(f"[Done] Index CSV -> {OUT_INDEX_CSV}")

if __name__ == "__main__":
    main()
