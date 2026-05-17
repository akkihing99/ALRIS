#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import argparse, os, json, warnings, shutil
from typing import List, Dict
from collections import defaultdict
from datetime import timedelta

import cv2, torch, numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

# Grounding-DINO
from model.GroundingDINO.groundingdino.models import build_model as build_dino
from model.GroundingDINO.groundingdino.util.slconfig import SLConfig
from model.GroundingDINO.groundingdino.util.utils import clean_state_dict
from model.GroundingDINO.groundingdino.util import box_ops
from model.GroundingDINO.groundingdino.util.inference import load_image, predict

# SAM
from segment_anything import build_sam, SamPredictor

import utils.config as config

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# CLASSES = [
#     "person", "sneakers", "chair", "other shoes", "hat",
#     "car", "lamp", "glasses", "bottle", "desk",
#     "cup", "street lights", "cabinet/shelf", "handbag/satchel", "bracelet",
#     "plate", "picture/frame", "helmet", "book", "gloves",
#     "storage box", "boat", "leather shoes", "flower", "bench",
#     "potted plant", "bowl/basin", "flag", "pillow", "boots",
#     "vase", "microphone", "necklace", "ring", "suv",
#     "wine glass", "belt", "monitor/tv", "backpack", "umbrella",
#     "traffic light", "speaker", "watch", "tie", "trash bin can",
#     "slippers", "bicycle", "stool", "barrel/bucket", "van",
#     "couch", "sandals", "basket", "drum", "pen/pencil",
#     "bus", "wild bird", "high heels", "motorcycle", "guitar",
#     "carpet", "cell phone", "bread", "camera", "canned",
#     "truck", "traffic cone", "cymbal", "lifesaver", "towel",
#     "stuffed toy", "candle", "sailboat", "laptop", "awning",
#     "bed", "faucet", "tent", "horse", "mirror",
#     "power outlet", "sink", "apple", "air conditioner", "knife",
#     "hockey stick", "paddle", "pickup truck", "fork", "traffic sign",
#     "balloon", "tripod", "dog", "spoon", "clock",
#     "pot", "cow", "cake", "dining table", "sheep",
#     "hanger", "blackboard/whiteboard", "napkin", "other fish", "orange/tangerine",
#     "toiletry", "keyboard", "tomato", "lantern", "machinery vehicle",
#     "fan", "green vegetables", "banana", "baseball glove", "airplane",
#     "mouse", "train", "pumpkin", "soccer", "skiboard",
#     "luggage", "nightstand", "tea pot", "telephone", "trolley",
#     "head phone", "sports car", "stop sign", "dessert", "scooter",
#     "stroller", "crane", "remote", "refrigerator", "oven",
#     "lemon", "duck", "baseball bat", "surveillance camera", "cat",
#     "jug", "broccoli", "piano", "pizza", "elephant",
#     "skateboard", "surfboard", "gun", "skating and skiing shoes", "gas stove",
#     "donut", "bow tie", "carrot", "toilet", "kite",
#     "strawberry", "other balls", "shovel", "pepper", "computer box",
#     "toilet paper", "cleaning products", "chopsticks", "microwave", "pigeon",
#     "baseball", "cutting/chopping board", "coffee table", "side table", "scissors",
#     "marker", "pie", "ladder", "snowboard", "cookies",
#     "radiator", "fire hydrant", "basketball", "zebra", "grape",
#     "giraffe", "potato", "sausage", "tricycle", "violin",
#     "egg", "fire extinguisher", "candy", "fire truck", "billiards",
#     "converter", "bathtub", "wheelchair", "golf club", "briefcase",
#     "cucumber", "cigar/cigarette", "paint brush", "pear", "heavy truck",
#     "hamburger", "extractor", "extension cord", "tong", "tennis racket",
#     "folder", "american football", "earphone", "mask", "kettle",
#     "tennis", "ship", "swing", "coffee machine", "slide",
#     "carriage", "onion", "green beans", "projector", "frisbee",
#     "washing machine/drying machine", "chicken", "printer", "watermelon", "saxophone",
#     "tissue", "toothbrush", "ice cream", "hot-air balloon", "cello",
#     "french fries", "scale", "trophy", "cabbage", "hot dog",
#     "blender", "peach", "rice", "wallet/purse", "volleyball",
#     "deer", "goose", "tape", "tablet", "cosmetics",
#     "trumpet", "pineapple", "golf ball", "ambulance", "parking meter",
#     "mango", "key", "hurdle", "fishing rod", "medal",
#     "flute", "brush", "penguin", "megaphone", "corn",
#     "lettuce", "garlic", "swan", "helicopter", "green onion",
#     "sandwich", "nuts", "speed limit sign", "induction cooker", "broom",
#     "trombone", "plum", "rickshaw", "goldfish", "kiwi fruit",
#     "router/modem", "poker card", "toaster", "shrimp", "sushi",
#     "cheese", "notepaper", "cherry", "pliers", "cd",
#     "pasta", "hammer", "cue", "avocado", "hami melon",
#     "flask", "mushroom", "screwdriver", "soap", "recorder",
#     "bear", "eggplant", "board eraser", "coconut", "tape measure/ruler",
#     "pig", "showerhead", "globe", "chips", "steak",
#     "crosswalk sign", "stapler", "camel", "formula 1", "pomegranate",
#     "dishwasher", "crab", "hoverboard", "meatball", "rice cooker",
#     "tuba", "calculator", "papaya", "antelope", "parrot",
#     "seal", "butterfly", "dumbbell", "donkey", "lion",
#     "urinal", "dolphin", "electric drill", "hair dryer", "egg tart",
#     "jellyfish", "treadmill", "lighter", "grapefruit", "game board",
#     "mop", "radish", "baozi", "target", "french",
#     "spring rolls", "monkey", "rabbit", "pencil case", "yak",
#     "red cabbage", "binoculars", "asparagus", "barbell", "scallop",
#     "noddles", "comb", "dumpling", "oyster", "table tennis paddle",
#     "cosmetics brush/eyeliner pencil", "chainsaw", "eraser", "lobster", "durian",
#     "okra", "lipstick", "cosmetics mirror", "curling", "table tennis"
# ]

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# CLASSES = [
#     # Living beings
#     "person", "animal", "pet", "bird", "mammal", "sea creature", "insect",
    
#     # Transportation
#     "vehicle", "car", "truck", "motorcycle", "bicycle", "aircraft", "watercraft",
    
#     # Indoor objects
#     "furniture", "appliance", "electronics", "kitchenware", "decoration",
    
#     # Personal items  
#     "clothing", "footwear", "accessories", "bag", "jewelry",
    
#     # Food & drink
#     "food", "fruit", "vegetable", "meal", "snack", "beverage", "ingredient",
    
#     # Tools & equipment
#     "tool", "utensil", "sports equipment", "musical instrument", "device",
    
#     # Outdoor & structures
#     "outdoor object", "sign", "light", "plant", "building element",
    
#     # General
#     "object", "container", "item", "product", "material"]
# CLASSES = [
#     # 1-5: aquatic mammals
#     'beaver',
#     'dolphin',
#     'otter',
#     'seal',
#     'whale',

#     # 6-10: fish
#     'aquarium_fish',
#     'flatfish',
#     'ray',
#     'shark',
#     'trout',

#     # 11-15: flowers
#     'orchids',
#     'poppies',
#     'roses',
#     'sunflowers',
#     'tulips',

#     # 16-20: household furniture (actually tableware/containers)
#     'bottles',
#     'bowls',
#     'cans',
#     'cups',
#     'plates',

#     # 21-25: fruit and vegetables
#     'apples',
#     'mushrooms',
#     'oranges',
#     'pears',
#     'sweet_peppers',

#     # 26-30: household electrical devices
#     'clock',
#     'computer_keyboard',
#     'lamp',
#     'telephone',
#     'television',

#     # 31-35: household furniture
#     'bed',
#     'chair',
#     'couch',
#     'table',
#     'wardrobe',

#     # 36-40: insects
#     'bee',
#     'beetle',
#     'butterfly',
#     'caterpillar',
#     'cockroach',

#     # 41-45: large carnivores
#     'bear',
#     'leopard',
#     'lion',
#     'tiger',
#     'wolf',

#     # 46-50: non-aquatic structures
#     'bridge',
#     'castle',
#     'house',
#     'road',
#     'skyscraper',

#     # 51-55: natural outdoor scenes
#     'cloud',
#     'forest',
#     'mountain',
#     'plain',
#     'sea',

#     # 56-60: large omnivores and herbivores
#     'camel',
#     'cattle',
#     'chimpanzee',
#     'elephant',
#     'kangaroo',

#     # 61-65: small mammals
#     'fox',
#     'porcupine',
#     'possum',
#     'raccoon',
#     'skunk',

#     # 66-70: invertebrates (non-insect)
#     'crab',
#     'lobster',
#     'snail',
#     'spider',
#     'worm',

#     # 71-75: people
#     'baby',
#     'boy',
#     'girl',
#     'man',
#     'woman',

#     # 76-80: reptiles
#     'crocodile',
#     'dinosaur',
#     'lizard',
#     'snake',
#     'turtle',

#     # 81-85: small mammals (alternate grouping)
#     'hamster',
#     'mouse',
#     'rabbit',
#     'shrew',
#     'squirrel',

#     # 86-90: trees
#     'maple',
#     'oak',
#     'palm',
#     'pine',
#     'willow',

#     # 91-95: vehicles (large passenger/utility)
#     'bicycle',
#     'bus',
#     'motorcycle',
#     'pickup_truck',
#     'train',

#     # 96-100: vehicles (non-passenger/specialized)
#     'lawn_mower',
#     'rocket',
#     'streetcar',
#     'tank',
#     'tractor'
# ]

GINO_PROMPT = ".".join(CLASSES)

# ------------- args -------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--box_th", type=float, default=0.1)
    p.add_argument("--text_th", type=float, default=0.1)
    p.add_argument("--max_masks", type=int, default=5)
    p.add_argument("--class_prompt", type=str, default=None,
                   help="custom prompt (falls back to GINO_PROMPT if unset)")
    p.add_argument("--dataset", type=str, default='phrasecut', help="Override dataset name in config")
    p.add_argument("--no_overwrite", action="store_true")
    p.add_argument("--save_empty", action="store_true")
    p.add_argument("--iou_thr", type=float, default=0.01)
    p.add_argument("--mask_path", type=str, default="datasets")
    return p.parse_args()

# ------------- DDP -------------
def ddp_env():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return True, torch.distributed.get_rank(), torch.distributed.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    return False, 0, 1, 0

def ddp_setup():
    is_dist, rank, world, local_rank = ddp_env()
    if is_dist and not torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(hours=5),
        )
    return ddp_env()

def is_main():
    is_dist, rank, *_ = ddp_env()
    return (not is_dist) or (rank == 0)

# ------------- utils -------------
def makedirs(p: str): os.makedirs(p, exist_ok=True)

def save_mask_png(path: str, mask_bool: np.ndarray):
    makedirs(os.path.dirname(path))
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)

def load_dino(cfg_path: str, ckpt: str, device: str):
    cfg = SLConfig.fromfile(cfg_path); cfg.device = device
    model = build_dino(cfg)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    model.eval().to(device)
    return model

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0: return 0.0
    union = np.logical_or(a, b).sum() + 1e-6
    return float(inter / union)

def allowed_by_iou(m: np.ndarray, selected: List[np.ndarray], thr: float) -> bool:
    return all(mask_iou(m, s) < thr for s in selected)

# ------------- simple dataset: unique image list from ann -------------
class ImageList(torch.utils.data.Dataset):
    def __init__(self, img_names: List[str]):
        self.img_names = img_names
    def __len__(self): return len(self.img_names)
    def __getitem__(self, idx): return self.img_names[idx]

# ------------- main -------------
@torch.no_grad()
def main():
    args = get_args()
    is_dist, rank, world, local_rank = ddp_setup()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.dataset:
        cfg.dataset = args.dataset
        if is_main():
            logger.info(f"Overriding dataset from args: {cfg.dataset}")
    if cfg.dataset == "phrasecut":
        img_root = "datasets/phrasecut/VGPhraseCut_v0/images"
    else:
        img_root = "datasets/images/train2014"

    ann_path = os.path.join("datasets", "anns", str(cfg.dataset), "train.json")

    mask_root_v1 = os.path.join(args.mask_path, "gsam_mask", str(cfg.dataset))
    mask_root_v2 = os.path.join(args.mask_path, "gsam_mask", f"{cfg.dataset}2")
    meta_dir     = os.path.join(args.mask_path, "meta")
    meta_v1_path = os.path.join(meta_dir,  f"gsam_mask_meta_{cfg.dataset}.json")
    stats_path   = os.path.join(meta_dir,  f"gsam_mask_stats_{cfg.dataset}.txt")
    meta_v2_path = os.path.join(meta_dir,  f"gsam_mask_meta_{cfg.dataset}2.json")

    if is_main():
        makedirs(mask_root_v1); makedirs(meta_dir)
        logger.info(f"DATASET: {cfg.dataset}")
        logger.info(f"IMG ROOT: {img_root}")
        logger.info(f"ANN PATH: {ann_path}")
    if is_dist: torch.distributed.barrier()

    # load unique image list from ann
    with open(ann_path, "r") as f:
        anns = json.load(f)
    # anns: list of dicts with "img_name"
    uniq = sorted({a["img_name"] for a in anns})
    if is_main():
        logger.info(f"Unique images in ann: {len(uniq)}")

    # dataset / sampler
    ds = ImageList(uniq)
    if is_dist:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, sampler=sampler,
        num_workers=cfg.workers, pin_memory=True, drop_last=False
    )

    # models
    dino = load_dino(
        "model/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "pretrain/groundingdino_swint_ogc.pth",
        device,
    )
    sam  = build_sam(checkpoint="pretrain/sam_vit_h_4b8939.pth").to(device)
    predictor = SamPredictor(sam)

    # prompt
    prompt = args.class_prompt.strip() if args.class_prompt else GINO_PROMPT
    if is_main():
        logger.info(f"Using prompt ({len(prompt)} chars): {prompt[:120]}...")

    # local accumulators
    meta_local: Dict[str, List[int]] = {}
    dist_local = defaultdict(int)
    total_images_local = 0
    total_masks_local  = 0

    pbar = tqdm(loader, desc=f"[rank {rank}] DINO➜SAM", ncols=100, disable=not is_main())
    for (img_name,) in pbar:
        img_name = img_name  # string
        img_base = os.path.splitext(os.path.basename(img_name))[0]
        save_dir = os.path.join(mask_root_v1, img_base)

        # no_overwrite: skip if folder exists
        if args.no_overwrite and os.path.isdir(save_dir):
            logger.info(f"[rank {rank}] skip existing: {img_name}")
            total_images_local += 1
            dist_local[len(os.listdir(save_dir)) if os.path.isdir(save_dir) else 0] += 1
            continue

        # load image
        img_path = os.path.join(img_root, img_name)
        try:
            image_source, image = load_image(img_path)
        except Exception as e:
            logger.warning(f"[{img_name}] load_image error: {e}")
            total_images_local += 1
            dist_local[0] += 1
            continue

        H, W, _ = image_source.shape
        predictor.set_image(image_source)

        # Grounding-DINO predict (multi-class prompt)
        try:
            boxes_cxcywh, logits, _ = predict(
                model=dino, image=image, caption=prompt,
                box_threshold=args.box_th, text_threshold=args.text_th, device=device
            )
        except Exception as e:
            logger.warning(f"[{img_name}] DINO error: {e}")
            total_images_local += 1
            dist_local[0] += 1
            continue

        # ---- Grounding-DINO -> SAM with fallback(top-1 @ box_th=0.05) ----
        keep_idx: List[int] = []

        def dino_predict(th):
            try:
                b, l, _ = predict(
                    model=dino, image=image, caption=prompt,
                    box_threshold=th, text_threshold=args.text_th, device=device
                )
                return b, l
            except Exception as e:
                logger.warning(f"[{img_name}] DINO error(th={th}): {e}")
                return None, None

        boxes_cxcywh, logits = dino_predict(args.box_th)

        fallback_used = False
        if boxes_cxcywh is None or boxes_cxcywh.shape[0] == 0:
            logger.warning(f"[{img_name}] No boxes at box_th={args.box_th}. Retrying with 0.05 (top-1 only).")
            b2, l2 = dino_predict(0.05)
            if b2 is None or b2.shape[0] == 0:
                total_images_local += 1
                dist_local[0] += 1
                if args.save_empty:
                    meta_local[img_base] = []
                continue
            top1 = torch.argmax(l2.flatten())
            boxes_cxcywh = b2[top1:top1+1]
            logits = l2[top1:top1+1]
            fallback_used = True

        logits = logits.flatten()
        if fallback_used:
            order = torch.tensor([0], device=logits.device)
        else:
            order = torch.argsort(logits, descending=True)
            order = order[:max(args.max_masks * 3, args.max_masks)]

        # to xyxy (px)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh[order]) * torch.tensor([W, H, W, H], device=logits.device)
        boxes_xyxy = boxes_xyxy.clamp(min=0, max=max(H, W))

        # SAM predict
        pred_device = next(predictor.model.parameters()).device
        boxes_in = predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        ).to(device=pred_device, dtype=torch.float32)

        try:
            masks, _, _ = predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=boxes_in, multimask_output=False
            )
            masks_np = masks[:, 0].detach().cpu().numpy().astype(bool)
        except Exception as e:
            logger.warning(f"[{img_name}] SAM error: {e}")
            masks_np = None

        if masks_np is not None:
            kept: List[np.ndarray] = []
            for k in range(masks_np.shape[0]):
                if fallback_used and len(kept) >= 1:
                    break
                if not fallback_used and len(kept) >= args.max_masks:
                    break
                m = masks_np[k]
                if m.sum() == 0:
                    continue
                if not allowed_by_iou(m, kept, args.iou_thr):
                    continue
                kept.append(m)

            if kept:
                for i, m in enumerate(kept):
                    save_mask_png(os.path.join(save_dir, f"{i}.png"), m)
                keep_idx = list(range(len(kept)))

        # meta/stat
        if keep_idx or args.save_empty:
            meta_local[img_base] = keep_idx
        total_images_local += 1
        total_masks_local  += len(keep_idx)
        dist_local[len(keep_idx)] += 1


    # gather
    if is_dist:
        torch.distributed.barrier()
        meta_all = [None for _ in range(world)]
        torch.distributed.all_gather_object(meta_all, meta_local)
        dist_all = [None for _ in range(world)]
        torch.distributed.all_gather_object(dist_all, dict(dist_local))
        totals_all = [None for _ in range(world)]
        torch.distributed.all_gather_object(totals_all, (total_images_local, total_masks_local))
    else:
        meta_all = [meta_local]
        dist_all = [dict(dist_local)]
        totals_all = [(total_images_local, total_masks_local)]

    # rank0: merge & save V1 + build V2
    if is_main():
        meta_v1: Dict[str, List[int]] = {}
        for m in meta_all:
            if m: meta_v1.update(m)

        dist_m = defaultdict(int)
        for d in dist_all:
            for k, v in (d or {}).items():
                dist_m[int(k)] += int(v)

        total_images = sum(t[0] for t in totals_all if t)
        total_masks  = sum(t[1] for t in totals_all if t)

        makedirs(meta_dir)
        with open(meta_v1_path, "w") as f:
            json.dump(meta_v1, f, indent=2, ensure_ascii=False)

        with open(stats_path, "w") as f:
            f.write(f"dataset: {cfg.dataset}\n")
            f.write(f"split: train\n")
            f.write(f"box_th: {args.box_th}\n")
            f.write(f"text_th: {args.text_th}\n")
            f.write(f"iou_thr: {args.iou_thr}\n")
            f.write(f"max_masks_per_image: {args.max_masks}\n")
            f.write(f"world_size: {world}\n")
            f.write(f"total_images_processed: {total_images}\n")
            f.write(f"total_masks_generated: {total_masks}\n")
            f.write("mask_count_distribution_per_image:\n")
            for k in sorted(dist_m.keys()):
                f.write(f"  {k}: {dist_m[k]}\n")
        logger.info(f"✔ V1 meta:  {meta_v1_path}")
        logger.info(f"✔ stats  :  {stats_path}")

        # ---- V2: flatten from V1 ----
        makedirs(mask_root_v2)
        meta_v2: Dict[str, List[str]] = {}
        gid = 0

        with open(meta_v1_path, "r") as f:
            meta_v1 = json.load(f)

        missing_files = 0
        for img_base in sorted(meta_v1.keys()):
            idxs = sorted(int(x) for x in meta_v1[img_base])
            assigned = []
            for k in idxs:
                src = os.path.join(mask_root_v1, img_base, f"{k}.png")
                if not os.path.isfile(src):
                    missing_files += 1
                    continue
                dst_name = f"{gid}.png"
                dst = os.path.join(mask_root_v2, dst_name)
                shutil.copyfile(src, dst)
                assigned.append(dst_name)
                gid += 1
            if assigned or args.save_empty:
                meta_v2[img_base] = assigned

        with open(meta_v2_path, "w") as f:
            json.dump(meta_v2, f, indent=2, ensure_ascii=False)

        logger.info(f"✔ V2 meta:  {meta_v2_path}")
        logger.info(f"✔ V2 masks: {mask_root_v2} (total={gid}, missing_from_v1={missing_files})")

        v1_total = sum(len(v) for v in meta_v1.values())
        v2_total = sum(len(v) for v in meta_v2.values())
        if v1_total != v2_total:
            logger.warning(f"[SANITY] V1(meta)={v1_total} != V2(meta)={v2_total} "
                        f"(missing_from_v1_files={missing_files})")
        else:
            logger.info(f"[SANITY] V1(meta) == V2(meta) == {v1_total}")

    if is_dist:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
