import argparse
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Make pseudo-to-GT matching")
    parser.add_argument("--dataset", default="refcoco", help="dataset name")
    return parser.parse_args()

args = parse_args()
DATASET = args.dataset.lower()

META_JSON = f"datasets/sam_gt_mask/meta_{DATASET}_train.json"
REFCOCO_META_JSON = f"datasets/meta/{DATASET}_meta.json"
PSEUDO_V2_ROOT = Path(f"datasets3/gsam_mask/{DATASET}2")                  # e.g., 0.png, 1.png, ...
PSEUDO_V2_META = Path(f"datasets3/meta/gsam_mask_meta_{DATASET}2.json")    # {"img_base":["12.png","13.png",...]}
GT_ROOT = Path(f"datasets/masks/{DATASET}")                               # {segment_id}.png

OUT_DIR = Path("datasets3/meta")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON_IDX = OUT_DIR / f"pseudo_to_gt_{DATASET}.json"          # {int -> gt_idx or -1}

IOU_THRESHOLD = 0.10

def load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 0)

def resize_to(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    pil = pil.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return (np.array(pil) > 0)

def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def parse_image_id_from_filename(filename: str) -> str:
    """
    COCO_train2014_000000581857.jpg -> '581857'
    """
    m = re.search(r'(\d+)(?=\.\w+$)', filename)
    if not m:
        raise ValueError(f"Could not parse numeric id from: {filename}")
    return str(int(m.group(1)))  # strip leading zeros

def load_meta_records(meta_path: str):
    with open(meta_path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError(f"Unexpected meta format in {meta_path}")

def collect_gt_masks_by_segment_ids(seg_ids: List[int]) -> List[Optional[np.ndarray]]:
    masks: List[Optional[np.ndarray]] = []
    for sid in seg_ids:
        path = GT_ROOT / f"{sid}.png"
        masks.append(load_mask(path))
    return masks

def build_iou_matrix(pseudo_masks: List[np.ndarray], gt_masks: List[Optional[np.ndarray]]) -> np.ndarray:
    P = len(pseudo_masks)
    G = len(gt_masks)
    if P == 0 or G == 0:
        return np.zeros((P, G), dtype=np.float32)
    H, W = pseudo_masks[0].shape
    mat = np.zeros((P, G), dtype=np.float32)
    for i, pm in enumerate(pseudo_masks):
        for j, gm in enumerate(gt_masks):
            if gm is None:
                mat[i, j] = 0.0
            else:
                gm_r = resize_to(gm, (H, W))
                mat[i, j] = iou_binary(pm, gm_r)
    return mat

def greedy_unique_match(iou_mat: np.ndarray, thr: float) -> Dict[int, int]:
    if iou_mat.size == 0:
        return {}
    P, G = iou_mat.shape
    pairs = []
    flat_idx = np.argsort(iou_mat.ravel())[::-1]
    used_p, used_g = set(), set()
    for k in flat_idx:
        i = k // G
        j = k % G
        if i in used_p or j in used_g:
            continue
        iou = float(iou_mat[i, j])
        if iou < thr:
            break
        pairs.append((i, j))
        used_p.add(i)
        used_g.add(j)
        if len(used_p) == P or len(used_g) == G:
            break
    return {i: j for (i, j) in pairs}

def main():
    if not PSEUDO_V2_META.exists():
        raise FileNotFoundError(f"Missing pseudo meta(v2): {PSEUDO_V2_META}")
    with open(PSEUDO_V2_META, "r") as f:
        pseudo_v2_meta: Dict[str, List[str]] = json.load(f)  # {"img_base":["12.png","13.png", ...]}

    records = load_meta_records(META_JSON)
    with open(REFCOCO_META_JSON, "r") as f:
        refcoco_meta: Dict[str, List[int]] = json.load(f)  # {"img_name":[gt_global_idx,...]}

    out_map_idx: Dict[int, int] = {}   # pseudo_global_idx -> gt_global_idx or -1

    total_pseudo_all = 0
    matched_all = 0

    seen_global_indices = set()

    for rec in tqdm(records):
        img_name = rec["img_name"]
        try:
            img_id = parse_image_id_from_filename(img_name)  # "581857"
        except Exception as e:
            print(f"[WARN] Skip {img_name}: {e}")
            continue

        img_base = img_name.rsplit(".", 1)[0]  # "COCO_train2014_000000581857"
        if img_base not in pseudo_v2_meta:
            seg_ids = [int(s["segment_id"]) for s in rec.get("segments", [])]
            num_pseudo = 0
            continue

        pseudo_files = pseudo_v2_meta[img_base]
        seg_ids = [int(s["segment_id"]) for s in rec.get("segments", [])]
        gt_index_list = refcoco_meta.get(img_name, [])  # [gt_global_idx, ...]

        if len(gt_index_list) != len(seg_ids):
            print(f"[WARN] {img_name}: refcoco_meta length({len(gt_index_list)}) != segments length({len(seg_ids)})")

        gt_masks = collect_gt_masks_by_segment_ids(seg_ids)

        pseudo_masks: List[Optional[np.ndarray]] = []
        pseudo_global_indices: List[int] = []
        local_keys: List[str] = []

        for local_idx, fname in enumerate(pseudo_files):
            try:
                gidx = int(Path(fname).stem)
            except Exception:
                continue

            if gidx in seen_global_indices:
                continue

            mask_path = PSEUDO_V2_ROOT / fname
            m = load_mask(mask_path)
            pseudo_masks.append(m)
            pseudo_global_indices.append(gidx)
            local_keys.append(f"{img_id}_{local_idx}")

        P = len(pseudo_masks)
        G = len(gt_masks)
        total_pseudo_all += P

        if P == 0:
            continue

        valid_pseudo_idx = [i for i, m in enumerate(pseudo_masks) if m is not None]
        if len(valid_pseudo_idx) == 0 or G == 0:
            for ii in range(P):
                gidx = pseudo_global_indices[ii]
                key_str = local_keys[ii]
                out_map_idx[gidx] = -1
                seen_global_indices.add(gidx)
            continue

        pm_list = [pseudo_masks[i] for i in valid_pseudo_idx]
        iou_mat = build_iou_matrix(pm_list, gt_masks)  # (len(valid_pseudo_idx), G)

        match_local = greedy_unique_match(iou_mat, IOU_THRESHOLD)  # local(valid-space) -> gt_local

        matched_set = set()
        valid_set = set(valid_pseudo_idx)

        for local_in_valid, j_local in match_local.items():
            p_local = valid_pseudo_idx[local_in_valid]
            gidx = pseudo_global_indices[p_local]
            key_str = local_keys[p_local]

            if j_local < len(gt_index_list):
                gt_global = int(gt_index_list[j_local])
            else:
                gt_global = -1

            out_map_idx[gidx] = gt_global
            seen_global_indices.add(gidx)
            if gt_global != -1:
                matched_set.add(p_local)

        for local_in_valid in range(len(valid_pseudo_idx)):
            p_local = valid_pseudo_idx[local_in_valid]
            if p_local in matched_set:
                continue
            gidx = pseudo_global_indices[p_local]
            key_str = local_keys[p_local]
            if gidx in out_map_idx:
                continue
            out_map_idx[gidx] = -1
            seen_global_indices.add(gidx)

        for ii in range(P):
            if ii in valid_set:
                continue
            gidx = pseudo_global_indices[ii]
            key_str = local_keys[ii]
            out_map_idx[gidx] = -1
            seen_global_indices.add(gidx)

        matched_all += len(matched_set)

    with open(OUT_JSON_IDX, "w") as f:
        json.dump(out_map_idx, f, indent=2, ensure_ascii=False)


    total_pseudo = len(out_map_idx)
    matched_vals = [v for v in out_map_idx.values() if v != -1]
    matched_count = len(matched_vals)

    cnt = Counter(matched_vals)
    dup_pseudo_count = sum(c for c in cnt.values() if c >= 2)

    matched_ratio_total = matched_count / total_pseudo if total_pseudo else 0.0
    dup_ratio_total = dup_pseudo_count / total_pseudo if total_pseudo else 0.0
    dup_ratio_among_matched = (dup_pseudo_count / matched_count) if matched_count else 0.0

    print("======================================")
    print(f"Saved (index map): {OUT_JSON_IDX}")
    print(f"Total pseudo masks: {total_pseudo}")
    print(f"Matched (IoU ≥ {IOU_THRESHOLD}): {matched_count}  "
          f"({matched_ratio_total*100:.2f}% of all)")
    print(f"GT-id duplicates among matched (for info):")
    print(f"  - Pseudos in duplicated groups: {dup_pseudo_count}")
    print(f"  - Ratio over ALL pseudos:       {dup_ratio_total*100:.2f}%")
    print(f"  - Ratio over MATCHED pseudos:   {dup_ratio_among_matched*100:.2f}%")
    print("Note: value = -1 means 'no GT above IoU threshold' (excluded).")
    print("======================================")

if __name__ == "__main__":
    main()
