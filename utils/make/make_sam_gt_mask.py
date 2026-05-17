# make_sam_bestmask.py
# ──────────────────────────────────────────────────────────

from __future__ import annotations
import argparse, os, warnings, json
import cv2, torch, numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

import utils.config as config
from utils.dataset_analysis import RefDataset
from utils.misc import worker_init_fn
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# ──────────────── args ────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--out_dir",   default="./datasets/sam_gt_mask/refcoco+")
    p.add_argument("--sam_ckpt",  default="pretrain/sam_vit_h_4b8939.pth")
    p.add_argument("--sam_type",  default="vit_h")
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--save_json", action="store_true")
    p.add_argument("--only_ids",  nargs="+", type=int, default=None,
                   help="Only process these seg_ids (optional)")
    return p.parse_args()

# ──────────────── util ────────────────────────────────────

def save_mask(path: str, mask_bool: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(mask_bool.astype(np.uint8) * 255).save(path)

# ──────────────── retry schedule ──────────────────────────

RETRY_PARAMS = [
    dict(points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95,
         crop_n_layers=0, crop_n_points_downscale_factor=1, min_mask_region_area=0),
    dict(points_per_side=32, pred_iou_thresh=0.78, stability_score_thresh=0.85,
         crop_n_layers=0, crop_n_points_downscale_factor=1, min_mask_region_area=10),
    dict(points_per_side=64, pred_iou_thresh=0.7, stability_score_thresh=0.8,
         crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=5),
    dict(points_per_side=64, pred_iou_thresh=0.6, stability_score_thresh=0.7,
         crop_n_layers=1, crop_n_points_downscale_factor=1, min_mask_region_area=0),
    dict(points_per_side=64, pred_iou_thresh=0.5, stability_score_thresh=0.6,
         crop_n_layers=1, crop_n_points_downscale_factor=1, min_mask_region_area=0),
]
# ──────────────── main ────────────────────────────────────

@torch.no_grad()
def main():
    args = get_args()
    cfg = config.load_cfg_from_cfg_file(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    initial_only_ids = set(args.only_ids) if args.only_ids else None

    ds = RefDataset(
        lmdb_dir   = cfg.train_lmdb,
        mask_dir   = cfg.mask_root_gt,
        dataset    = cfg.dataset,
        split      = cfg.train_split,
        mode       = "test",
        input_size = cfg.input_size,
        word_length= cfg.word_len,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False,
        worker_init_fn=lambda w: worker_init_fn(w, args.workers, 0, 42))

    # ── SAM backbone ────────────────────────────────────────────────
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    sam.to(device)

    #  iterate over RETRY_PARAMS schedule
    remained_ids = set(initial_only_ids) if initial_only_ids else None  # None => all
    overall_iou, saved_cnt = [], 0
    
    for trial, sam_kwargs in enumerate(RETRY_PARAMS, start=1):
        if remained_ids is not None and len(remained_ids) == 0:
            logger.info(f"✅ All masks generated. Early stopping at trial {trial - 1}")
            break

        logger.info(f"[Trial {trial}] Building SAM mask generator with params: {sam_kwargs}")
        mask_gen = SamAutomaticMaskGenerator(sam, **sam_kwargs)

        trial_remained = set()

        for _, params in tqdm(loader, ncols=100, desc=f"SAM‑GT (trial {trial})"):
            seg_id = int(params["seg_id"][0])
            if remained_ids is not None and seg_id not in remained_ids:
                continue

            ori_img = params["ori_img"][0].numpy()[..., ::-1]  # RGB→BGR
            gt_path = params["mask_dir"][0]
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                logger.warning(f"[{seg_id}] GT mask missing – skip");  trial_remained.add(seg_id);  continue
            gt_bool = gt_mask.astype(bool)

            # SAM mask list
            masks = mask_gen.generate(ori_img[..., ::-1])  # RGB
            if not masks:
                trial_remained.add(seg_id);  continue

            best_iou, best_mask = 0.0, None
            for m in masks:
                pred_bool = m["segmentation"].astype(bool)
                inter = np.logical_and(pred_bool, gt_bool).sum()
                union = np.logical_or(pred_bool, gt_bool).sum() + 1e-6
                iou = inter / union
                if iou > best_iou:
                    best_iou, best_mask = iou, pred_bool

            if best_mask is None:
                trial_remained.add(seg_id);  continue

            save_mask(os.path.join(args.out_dir, f"{seg_id}.png"), best_mask)
            overall_iou.append(best_iou);  saved_cnt += 1

        remained_ids = trial_remained if (initial_only_ids or trial_remained) else None
        logger.info(f"Trial {trial} done →  saved accum={saved_cnt},  remained={len(remained_ids) if remained_ids else 0}")

    if overall_iou:
        mIoU = float(np.mean(overall_iou))
        logger.success(f"◎ total_saved={saved_cnt}  total_skipped={len(remained_ids) if remained_ids else 0}  mIoU={mIoU:.4f}")
        if args.save_json:
            out_json = os.path.join(args.out_dir, "sam_iou_stats.json")
            json.dump({"mean_iou": mIoU, "per_sample": overall_iou}, open(out_json, "w"), indent=2)
            logger.info(f"IoU stats saved ➜ {out_json}")
    else:
        logger.warning("No mask saved – check dataset / SAM parameters.")

    if remained_ids:
        logger.warning("✖ Following seg_id(s) could not generate a mask even after 5 trials:")
        logger.warning(sorted(list(remained_ids)))


if __name__ == "__main__":
    main()
