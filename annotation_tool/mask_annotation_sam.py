import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pyarrow")
import sys, os, json, argparse, cv2, numpy as np, tqdm, torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler
from shapely.geometry import Polygon
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils.dataset_annotation import RefcocoMaskAnnotation
from PIL import Image
import alphashape
from rdp import rdp
import datetime 
# ------------------ DDP Setup ------------------
def ddp_setup():
    local = int(os.environ.get("LOCAL_RANK", -1))
    if local != -1:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=36000))
        torch.cuda.set_device(local)
    return (dist.get_rank()  if dist.is_initialized() else 0,
            dist.get_world_size() if dist.is_initialized() else 1)

rank, world = ddp_setup()
logger.info(f"[RANK {rank}] using CUDA device {torch.cuda.current_device()}")

# ------------------ Utility ------------------
def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum() + 1e-6
    return inter / union

def estimate_polygon_clicks(path: str, eps: float = 1.0) -> int:
    m = cv2.imread(path, 0)
    if m is None:
        return 0

    _, binary = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)

    approx = cv2.approxPolyDP(cnt, eps, True)  # ε in pixels
    if approx is None:
        return 0
    k = len(approx.reshape(-1, 2))
    return k if k >= 3 else 0

def mask_clicks_to_gt(gt_bool, init_mask, sam_masks, min_delta=0.01):
    base_iou = iou(init_mask, gt_bool)
    current = init_mask.copy()
    selected = set()
    clicks = 0

    candidate_idxs = set(range(len(sam_masks)))
    while True:
        best_gain, best_idx, best_tmp = -1, None, None
        for idx in candidate_idxs:
            cand = sam_masks[idx]["segmentation"]
            add_iou = iou(np.logical_or(current, cand), gt_bool)
            rem_iou = iou(np.logical_and(current, np.logical_not(cand)), gt_bool)
            if add_iou >= rem_iou:
                gain, tmp = add_iou - base_iou, np.logical_or(current, cand)
            else:
                gain, tmp = rem_iou - base_iou, np.logical_and(current, np.logical_not(cand))
            if gain > best_gain:
                best_gain, best_idx, best_tmp = gain, idx, tmp

        if best_gain >= min_delta:
            current = best_tmp
            base_iou += best_gain
            clicks += 1
            selected.add(best_idx)
            candidate_idxs.remove(best_idx)
        else:
            break

    return clicks, current, iou(init_mask, gt_bool), base_iou

def save_mask(path: str, mask_bool: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(mask_bool.astype(np.uint8) * 255).save(path)

# ------------------ SAM Config ------------------
SAM_CFG = dict(
    points_per_side=64,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=0,
)

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam_ckpt", default="pretrain/sam_vit_h_4b8939.pth")
    ap.add_argument("--sam_type", default="vit_h")
    ap.add_argument("--dataset", default="refcoco")
    args = ap.parse_args()

    dataset = args.dataset
    img_root = f"./datasets/images/train2014/"
    sam_root = f'datasets/sam_gt_mask/{dataset}'
    gt_root = f"./datasets/masks/{dataset}"
    lmdb = f"./datasets/lmdb/{dataset}/train.lmdb"

    save_correction_dir = f"./datasets/correction/{dataset}"
    save_scratch_dir = f"./datasets/from_scratch/{dataset}"
    os.makedirs(save_correction_dir, exist_ok=True)
    os.makedirs(save_scratch_dir, exist_ok=True)

    ds = RefcocoMaskAnnotation(dataset, lmdb, img_root, gt_root, sam_root)
    loader = DataLoader(ds, batch_size=1, num_workers=8,
        sampler=DistributedSampler(ds, world, rank, shuffle=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_net = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt).to(device)
    sam_gen = SamAutomaticMaskGenerator(sam_net, **SAM_CFG)

    stats_scratch, stats_correction = {}, {}

    for meta in tqdm.tqdm(loader, disable=(rank != 0)):
        idx = int(meta["index"][0])
        seg_id = meta["seg_id"][0]

        img_rgb = cv2.cvtColor(cv2.imread(os.path.join(img_root, meta["img_name"][0])), cv2.COLOR_BGR2RGB)
        gt_bool = (cv2.imread(meta["gt_mask_dir"][0], 0) > 127)
        sam_bool = (cv2.imread(meta['sam_mask_dir'][0], 0) > 127)

        masks = sam_gen.generate(img_rgb)
        if not masks: continue

        poly_clicks = estimate_polygon_clicks(meta["gt_mask_dir"][0])

        # Case 1: SAM correction
        c_clicks, c_mask, iou_ori, iou_corr = mask_clicks_to_gt(gt_bool, sam_bool, masks)
        stats_correction[idx] = {"poly": poly_clicks, "sp": c_clicks, "iou_sam": round(iou_ori, 4), "iou_corr": round(iou_corr, 4)}
        save_mask(os.path.join(save_correction_dir, f"{seg_id}.png"), c_mask)

        # Case 2: Scratch (from zero)
        s_clicks, s_mask, iou_zero, iou_final = mask_clicks_to_gt(gt_bool, np.zeros_like(gt_bool), masks)
        stats_scratch[idx] = {"poly": poly_clicks, "sp": s_clicks, "iou_zero": round(iou_zero, 4), "iou_corr": round(iou_final, 4)}
        save_mask(os.path.join(save_scratch_dir, f"{seg_id}.png"), s_mask)

    def reduce_and_save(name, stats_dict, iou_keys):
        if dist.is_initialized():
            dist.barrier()
            try:
                gathered = [None] * world
                dist.all_gather_object(gathered, stats_dict)
            except Exception as e:
                logger.error(f"[Rank {rank}] all_gather_object failed: {e}")
                return
            if rank == 0:
                merged = {}
                for g in gathered:
                    if g: merged.update(g)
        else:
            merged = stats_dict

        if rank == 0:
            out_dir = f"datasets/al_cost/mask/{dataset}"
            os.makedirs(out_dir, exist_ok=True)
            json.dump(merged, open(f"{out_dir}/{name}_stats.json", "w"), indent=2)

            n = len(merged)
            total_clicks = sum(d["sp"] for d in merged.values())
            total_poly = sum(d["poly"] for d in merged.values())
            avg_iou = [sum(d[k] for d in merged.values()) / n for k in iou_keys]

            with open(f"{out_dir}/{name}_summary.txt", "w") as f:
                f.write(f"=== Click Cost Summary: {name} ===\n")
                f.write(f"Total polygon clicks      : {total_poly:,}\n")
                f.write(f"Total SAM clicks          : {total_clicks:,}\n")
                f.write(f"Relative cost (SP / poly) : {total_clicks / total_poly:.3f}\n\n")
                for k, v in zip(iou_keys, avg_iou):
                    f.write(f"Mean {k}: {v:.4f}\n")

            logger.success(f"[{name}] JSON and summary saved.")

    reduce_and_save("sam_correction", stats_correction, ["iou_sam", "iou_corr"])
    reduce_and_save("sam_from_scratch", stats_scratch, ["iou_zero", "iou_corr"])

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
