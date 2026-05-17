from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from loguru import logger


__all__ = [
    "load_cost_tables_separate",
    "full_gt_cost_separate",
    "ActiveLearning",
    "RandomSampling",
    "RegionEntropy",
]


def load_data_meta(dataset: str):
    with open(f"datasets/meta/{dataset}_gsam_llava_meta.json", "r") as f:
        imgname2idx = json.load(f)
    idx2imgname = {}
    for img_name, idx_list in imgname2idx.items():
        for idx in idx_list:
            idx2imgname[int(idx)] = img_name
    return imgname2idx, idx2imgname


def full_gt_cost_separate(dataset: str) -> Dict[str, float]:
    d = dataset.lower()
    if d == "refcoco":
        return {"mask": 12208 / 2, "text": 12208 / 2}
    if d == "refcoco+":
        return {"mask": 1612709, "text": 6764771.01}
    if d == "refcocog_u":
        return {"mask": 1575975, "text": 9234643.34}
    raise ValueError(f"Unknown dataset for full_gt_cost_separate: {dataset}")


FIXED_TOTAL_COSTS = {
    "real_naive":      0.28789737,
    "real_ours_total": 0.16372040,
    "real_ours_human": 0.15924026,
}


def load_cost_tables_separate(
    dataset: str,
    mask_json: str | None = None,
    text_json: str | None = None,
    cost_key: str = "real_ours_total",
    use_first_only: bool = False,
):
    d = dataset.lower()

    if cost_key in FIXED_TOTAL_COSTS:
        target_val = FIXED_TOTAL_COSTS[cost_key] / 2.0
        ref_path = mask_json or f"datasets/al_cost/mask3/{d}/sam_correction_stats.json"
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Need an index reference file for fixed-cost mode: {ref_path}")
        with open(ref_path, "r") as f:
            ref_data = json.load(f)
        mask_cost = {int(k): target_val for k in ref_data.keys()}
        text_cost = {int(k): target_val for k in ref_data.keys()}
        return mask_cost, text_cost

    KEY_MAP = {
        "ours":               ("sp",   "our_3gram"),
        "naive":              ("poly", "typing_3gram"),
        "oursMask_typing":    ("sp",   "typing_3gram"),
        "poly_oursText":      ("poly", "our_3gram"),
    }
    if cost_key not in KEY_MAP:
        raise ValueError(
            f"Unknown cost_key {cost_key!r}. Available: "
            f"{list(FIXED_TOTAL_COSTS) + list(KEY_MAP)}")
    mask_key, text_key = KEY_MAP[cost_key]

    if mask_json is None:
        mask_json = f"datasets/al_cost/mask3/{d}/sam_correction_stats.json"
    if text_json is None:
        text_json = f"datasets/al_cost/text3/{d}/text_cost_5.json"
    if not os.path.exists(mask_json):
        raise FileNotFoundError(mask_json)
    if not os.path.exists(text_json):
        raise FileNotFoundError(text_json)

    with open(mask_json, "r") as f:
        raw_mask = json.load(f)
    mask_cost = {int(k): float(v.get(mask_key, 0.0)) for k, v in raw_mask.items()}

    with open(text_json, "r") as f:
        raw_text = json.load(f)
    text_cost: Dict[int, float] = {}
    for idx_str, per_seg in raw_text.items():
        idx = int(idx_str)
        total = 0.0
        if isinstance(per_seg, dict):
            if use_first_only:
                try:
                    first_blk = next(iter(per_seg.values()))
                    if isinstance(first_blk, dict):
                        total = float(first_blk.get(text_key, 0.0))
                except StopIteration:
                    total = 0.0
            else:
                for _, sent_blk in per_seg.items():
                    if isinstance(sent_blk, dict):
                        total += float(sent_blk.get(text_key, 0.0))
        text_cost[idx] = total
    return mask_cost, text_cost


class ActiveLearning:
    def __init__(self, args, dataset_size: int,
                 initial_labeled_ratio: float = 0.05, seed: int = 1234) -> None:
        if not 0.0 < initial_labeled_ratio <= 1.0:
            raise ValueError("initial_labeled_ratio must be in (0, 1].")
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        self.args = args
        self.dataset_size = dataset_size
        self._labeled_mask: List[bool] = [False] * dataset_size
        self._rng = random.Random(seed)
        self.pseudo2gt, self.gt2pseudos = self._load_pseudo2gt_index()

    def _load_pseudo2gt_index(self) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        d = self.args.dataset.lower()
        path = f"datasets/meta/pseudo_to_gt_{d}.json"
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Index mapping file not found: {path}\n"
                "Run annotation_tool/make_pseudo_to_gt.py to create it.")
        with open(path, "r") as f:
            raw = json.load(f)
        pseudo2gt = {int(k): int(v) for k, v in raw.items()}
        gt2pseudos: Dict[int, List[int]] = {}
        for p, g in pseudo2gt.items():
            if g == -1:
                continue
            gt2pseudos.setdefault(g, []).append(p)
        return pseudo2gt, gt2pseudos

    def labeled_indices(self)   -> List[int]: return [i for i, f in enumerate(self._labeled_mask) if f]
    def unlabeled_indices(self) -> List[int]: return [i for i, f in enumerate(self._labeled_mask) if not f]
    def set_model(self, model) -> None: pass
    def set_dataset(self, ds)   -> None: self.dataset = ds

    def update(self, newly_labeled: Sequence[int]) -> None:
        for idx in newly_labeled:
            if not 0 <= idx < self.dataset_size:
                raise IndexError(idx)
            self._labeled_mask[idx] = True

    def gt_index_of(self, idx: int) -> int:
        return int(self.pseudo2gt.get(idx, -1))

    def eligible_unlabeled_indices(self) -> List[int]:
        mp = self.pseudo2gt
        return [i for i in self.unlabeled_indices() if int(mp.get(i, -1)) != -1]

    def get_scores(self, indices: List[int] | None = None, round: int | None = None) -> Dict[int, float]:
        raise NotImplementedError

    def select_query_with_rejection(
        self,
        mask_cost: Dict[int, float],
        text_cost: Dict[int, float],
        round_idx: int,
        combined_budget: float,
        mask_budget: float | None = None,
        text_budget: float | None = None,
        reject_cost: float = 1.0,
    ):
        if combined_budget is None and (mask_budget is None or text_budget is None):
            raise ValueError("Provide either combined_budget or both mask_budget & text_budget.")

        candidate_pseudos = self.unlabeled_indices()
        if not candidate_pseudos:
            return set(), 0.0, 0.0, 0.0, {}

        logger.info(f"Calculating scores for {len(candidate_pseudos)} unlabeled pseudo samples...")
        scores = self.get_scores(candidate_pseudos, round_idx)
        ordered = sorted(candidate_pseudos, key=scores.get, reverse=True)

        selected, sel_scores = set(), {}
        spent = m_spent = t_spent = 0.0
        for p_idx in ordered:
            g_idx = self.gt_index_of(p_idx)
            if g_idx != -1:
                m_c = float(mask_cost.get(g_idx, 0.0))
                t_c = float(text_cost.get(g_idx, 0.0))
            else:
                m_c, t_c = reject_cost, 0.0
            c = m_c + t_c
            if combined_budget is not None:
                if spent + c > combined_budget:
                    break
            else:
                if m_spent + m_c > float(mask_budget) or t_spent + t_c > float(text_budget):
                    break
            selected.add(p_idx)
            sel_scores[p_idx] = scores.get(p_idx, 0.0)
            spent   += c
            m_spent += m_c
            t_spent += t_c
        return selected, spent, m_spent, t_spent, sel_scores

    def __len__(self):    return self.dataset_size
    def __repr__(self):
        return (f"{self.__class__.__name__}(dataset_size={self.dataset_size}, "
                f"labeled={len(self.labeled_indices())}, "
                f"unlabeled={len(self.unlabeled_indices())})")


class RandomSampling(ActiveLearning):
    def get_scores(self, indices: List[int] | None = None, round: int | None = None) -> Dict[int, float]:
        if indices is None:
            indices = self.unlabeled_indices()
        return {idx: random.random() for idx in indices}


class RegionEntropy(ActiveLearning):
    def __init__(self, dataset, batch_size: int = 64, num_workers: int = 1,
                 device: str = "cuda", repeats: int = 1, temp: float = 1.0,
                 intra_mode: str = "mean", pixel_mi: bool = False,
                 eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.bs, self.nw = int(batch_size), int(num_workers)
        self.dev = torch.device(device)
        self.repeats = int(max(1, repeats))
        self.eps = float(eps)
        self.temp = float(temp)
        self.intra_mode = str(intra_mode)
        self.pixel_mi = bool(pixel_mi)
        self.model = None
        self.img2idx, self.idx2img = load_data_meta(self.args.dataset)

    def set_model(self, m): self.model = m.eval()
    def set_dataset(self, d): self.dataset = d
    @staticmethod
    def _qnorm(x: torch.Tensor) -> torch.Tensor:
        if x.numel() <= 1:
            return torch.full_like(x, 0.5, dtype=torch.float32)
        vals, idx = torch.sort(x)
        r = torch.zeros_like(vals, dtype=torch.float32)
        r[idx] = torch.arange(len(x), device=x.device, dtype=torch.float32)
        return r / max(1e-9, float(len(x) - 1))

    def _H_multiclass_norm(self, p: np.ndarray) -> float:
        p = np.clip(p, self.eps, 1.0); p = p / p.sum()
        H = float(-(p * np.log(p)).sum())
        C = float(len(p))
        return H / max(np.log(C), self.eps)
    @torch.no_grad()
    def _idx_stats(self, pseudo_indices: List[int]) -> Tuple[torch.Tensor, np.ndarray, List[int]]:
        assert self.model is not None, "call set_model() first"
        if hasattr(self.dataset, "pseudo_row_of"):
            row_indices = [self.dataset.pseudo_row_of.get(p)
                           for p in pseudo_indices if p in self.dataset.pseudo_row_of]
            row_indices = [r for r in row_indices if r is not None]
            subset = Subset(self.dataset, row_indices)
        else:
            subset = Subset(self.dataset, pseudo_indices)

        feat_sum, cnt = {}, {}
        raw_logits_map: Dict[Tuple[int, int], torch.Tensor] = {}

        for i in range(self.repeats):
            if hasattr(self.dataset, "set_sent_idx"):
                self.dataset.set_sent_idx(i)
            loader = DataLoader(subset, self.bs, False, num_workers=self.nw, pin_memory=True)
            pbar = tqdm(loader, desc=f"Region-entropy pass {i+1}/{self.repeats}", ncols=80,
                        disable=(self.args.rank != 0))
            for img, txt, mask, ds_idx, *_ in pbar:
                img, txt = img.to(self.dev), txt.to(self.dev)
                logits, feats4d = self.model(img, txt, encoder_feats=True)
                gap_batch = F.normalize(feats4d.mean((2, 3)), p=2, dim=1).cpu()
                for b, pidx in enumerate(ds_idx.tolist()):
                    feat_sum[pidx] = feat_sum.get(pidx, 0) + gap_batch[b]
                    cnt[pidx] = cnt.get(pidx, 0) + 1
                    raw_logits_map[(pidx, i)] = logits[b].cpu()

        if hasattr(self.dataset, "set_sent_idx"):
            self.dataset.set_sent_idx(None)
        img_to_pidxs: Dict[str, List[int]] = defaultdict(list)
        for p_idx in pseudo_indices:
            if p_idx in self.idx2img:
                img_to_pidxs[self.idx2img[p_idx]].append(p_idx)

        prob_vectors_map = {}
        pbar_post = tqdm(img_to_pidxs.items(), desc="Region-entropy post-process",
                         ncols=80, disable=(self.args.rank != 0))
        for img_name, pidxs_in_image in pbar_post:
            mask_row_indices = [self.dataset.pseudo_row_of.get(p) for p in pidxs_in_image
                                if hasattr(self.dataset, "pseudo_row_of")
                                and p in self.dataset.pseudo_row_of] or pidxs_in_image
            all_masks_in_image = torch.stack(
                [self.dataset[row_idx][2] for row_idx in mask_row_indices]
            ).to(self.dev)

            for pidx in pidxs_in_image:
                target_mask_idx = pidxs_in_image.index(pidx)
                for r_i in range(self.repeats):
                    if (pidx, r_i) not in raw_logits_map:
                        continue
                    logits = raw_logits_map[(pidx, r_i)].to(self.dev)
                    z_l = logits.squeeze(0)            # (H, W)

                    obj_logits = []
                    for mask_tensor in all_masks_in_image:
                        w = F.adaptive_avg_pool2d(
                            mask_tensor.float().unsqueeze(0).unsqueeze(0), z_l.shape
                        ).squeeze(0).squeeze(0).clamp(0, 1)
                        wsum = w.sum()
                        z_obj = float(((z_l * w).sum() / (wsum + self.eps)).item()) if wsum > 0 else 0.0
                        obj_logits.append(z_obj)

                    union_mask = all_masks_in_image.max(dim=0)[0]
                    w_bg = (1.0 - F.adaptive_avg_pool2d(
                        union_mask.float().unsqueeze(0).unsqueeze(0), z_l.shape
                    ).squeeze(0).squeeze(0)).clamp(0, 1)
                    wsum_bg = w_bg.sum()
                    z_bg = float(((z_l * w_bg).sum() / (wsum_bg + self.eps)).item()) if wsum_bg > 0 else 0.0

                    logits_vec = np.asarray(obj_logits + [z_bg], dtype=np.float64)
                    z = (logits_vec - logits_vec.max()) / max(self.temp, self.eps)
                    p_vec = np.exp(z); p_vec /= p_vec.sum()
                    prob_vectors_map[(pidx, r_i)] = {
                        'probs': p_vec.astype(np.float32),
                        'target_idx': target_mask_idx,
                    }
        pidx_to_intra_score = {}
        for pidx in pseudo_indices:
            plist = [prob_vectors_map.get((pidx, i)) for i in range(self.repeats)
                     if (pidx, i) in prob_vectors_map]
            if not plist:
                continue
            P = np.stack([p['probs'] for p in plist], axis=0)
            p_bar = P.mean(axis=0)
            H_bar = self._H_multiclass_norm(p_bar)
            target_logits = [p['probs'][p['target_idx']] for p in plist]
            target_mean = float(np.mean(target_logits))
            score = H_bar / max(target_mean, self.eps)
            pidx_to_intra_score[pidx] = max(0.0, score)

        img_intra_scores = {
            name: np.mean([pidx_to_intra_score.get(p, 0.0) for p in pidxs])
            for name, pidxs in img_to_pidxs.items()
            if any(p in pidx_to_intra_score for p in pidxs)
        }

        feats, mi_arr, valid_indices = [], [], []
        for pidx in pseudo_indices:
            if pidx not in cnt:
                continue
            valid_indices.append(pidx)
            feats.append(F.normalize(feat_sum[pidx] / cnt[pidx], p=2, dim=0))
            img_name = self.idx2img.get(pidx)
            mi_arr.append(img_intra_scores.get(img_name, 0.0))
        return (torch.stack(feats) if feats else torch.empty(0, 0)), np.array(mi_arr, np.float32), valid_indices
    @torch.no_grad()
    def select_query_with_rejection(
        self,
        mask_cost: Dict[int, float],
        text_cost: Dict[int, float],
        round_idx: int,
        combined_budget: float,
        reject_cost: float = 1.0,
        **kwargs,
    ):
        unlabeled_pseudos = self.unlabeled_indices()
        if not unlabeled_pseudos or combined_budget <= 0:
            return set(), 0.0, 0.0, 0.0, {}

        _, mi_all, valid_indices = self._idx_stats(unlabeled_pseudos)
        if mi_all.size == 0:
            return set(), 0.0, 0.0, 0.0, {}
        mi_all = torch.from_numpy(mi_all).to(self.dev)

        p_idx_to_pos = {pid: i for i, pid in enumerate(valid_indices)}
        unlabeled_pos = [p_idx_to_pos[p] for p in unlabeled_pseudos if p in p_idx_to_pos]
        mi_u_t = mi_all[unlabeled_pos]

        img_to_pseudos: Dict[str, List[int]] = defaultdict(list)
        for p_idx in [valid_indices[pos] for pos in unlabeled_pos]:
            img_name = self.idx2img.get(p_idx)
            if img_name:
                img_to_pseudos[img_name].append(p_idx)

        img_names = sorted(img_to_pseudos.keys()); M = len(img_names)
        img_cost = torch.zeros(M, device=self.dev)
        for r, name in enumerate(img_names):
            csum = sum(
                (float(mask_cost.get(self.gt_index_of(p), 0)
                       + text_cost.get(self.gt_index_of(p), 0)))
                if self.gt_index_of(p) != -1 else reject_cost
                for p in img_to_pseudos[name]
            )
            img_cost[r] = csum

        row_ids_list = [r for r, name in enumerate(img_names) for _ in img_to_pseudos[name]]
        row_ids = torch.tensor(row_ids_list, device=self.dev, dtype=torch.long)
        mi_q_obj = self._qnorm(mi_u_t).clamp(0, 1)
        sum_intra = torch.zeros(M, device=self.dev).index_add_(0, row_ids, mi_q_obj)
        cnt_per_img = torch.zeros(M, device=self.dev).index_add_(0, row_ids, torch.ones_like(mi_q_obj))
        intra_q = sum_intra / cnt_per_img.clamp_min(1.0)

        sorted_indices = torch.argsort(intra_q, descending=True)
        chosen_pseudos: set = set()
        score_dict: Dict[int, float] = {}
        m_sp = t_sp = 0.0
        remain = float(combined_budget)
        for img_idx in sorted_indices.tolist():
            cost = img_cost[img_idx].item()
            if cost <= remain:
                remain -= cost
                name = img_names[img_idx]
                pseudos = img_to_pseudos[name]
                chosen_pseudos.update(pseudos)
                score_val = intra_q[img_idx].item()
                for p_idx in pseudos:
                    score_dict[p_idx] = score_val
                    g_idx = self.gt_index_of(p_idx)
                    if g_idx != -1:
                        m_sp += float(mask_cost.get(g_idx, 0))
                        t_sp += float(text_cost.get(g_idx, 0))
                    else:
                        m_sp += reject_cost
        spent = float(m_sp + t_sp)
        return chosen_pseudos, spent, float(m_sp), float(t_sp), score_dict
