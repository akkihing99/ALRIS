import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vip_llava')))
from datetime import timedelta
import os, json, argparse, re, collections, math, numpy as np, cv2, torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, DistributedSampler
import unicodedata
from utils.dataset_annotation import RefcocoTextAnnotation
from vip_llava.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vip_llava.llava.conversation import conv_templates
from vip_llava.llava.model.builder import load_pretrained_model
from vip_llava.llava.mm_utils import tokenizer_image_token

TOPK_SET   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
FALLBACK_H = 4.176
ALPHABET   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── DDP setup ───────────────────────────────────────────────
def ddp_setup():
    local = int(os.environ.get("LOCAL_RANK", -1))
    if local != -1:
        dist.init_process_group("nccl", timeout=timedelta(hours=12))
        torch.cuda.set_device(local)
    rank  = dist.get_rank()  if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device= f"cuda:{local}" if local != -1 else "cuda:0"
    return rank, world, device

rank, world, device = ddp_setup()

def normalize_chars_for_ngram(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z]+", "", s)
    return s

def load_counts_file(path: str, n: int):
    cnt = collections.Counter()
    if not path:
        return cnt
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token, last = parts[0], parts[-1]
            try:
                c = int(last)
            except:
                continue
            token = token.upper()
            if len(token) != n:
                continue
            if not all(ch in ALPHABET for ch in token):
                continue
            cnt[token] += c
    return cnt

def entropy_bits_from_monogram_counts(uni_cnt: collections.Counter) -> float:
    tot = sum(uni_cnt.values())
    if tot <= 0:
        return FALLBACK_H
    H = 0.0
    for a, c in uni_cnt.items():
        p = c / tot
        if p > 0:
            H += -p * math.log2(p)
    return H

class NGramModel:
    def __init__(self, uni_cnt, bi_cnt, tri_cnt):
        self.uni = uni_cnt
        self.bi  = bi_cnt
        self.tri = tri_cnt
        self.total_uni = sum(self.uni.values())
        self.bi_den  = collections.Counter()
        for bg, c in self.bi.items():
            self.bi_den[bg[0]] += c
        self.tri_den = collections.Counter()
        for tg, c in self.tri.items():
            self.tri_den[tg[:2]] += c

    def p1(self, x):
        num = self.uni.get(x, 0)
        if self.total_uni <= 0:
            return 1.0 / len(ALPHABET)
        if num == 0:
            return 1.0 / (self.total_uni * 1e6)
        return num / self.total_uni

    def p2_backoff(self, y, x):
        den = self.bi_den.get(x, 0)
        if den > 0:
            num = self.bi.get(x + y, 0)
            if num > 0:
                return num / den
        return self.p1(y)

    def p3_backoff(self, z, xy):
        den = self.tri_den.get(xy, 0)
        if den > 0:
            num = self.tri.get(xy + z, 0)
            if num > 0:
                return num / den
        return self.p2_backoff(z, xy[-1])

    def token_bits(self, segment_chars: str, order: int = 3) -> float:
        bits = 0.0
        ctx = ""
        for ch in segment_chars:
            if order >= 3 and len(ctx) >= 2:
                prob = self.p3_backoff(ch, ctx[-2:])
            elif order >= 2 and len(ctx) >= 1:
                prob = self.p2_backoff(ch, ctx[-1])
            else:
                prob = self.p1(ch)
            prob = max(prob, 1e-15)
            bits += -math.log2(prob)
            ctx += ch
        return bits

# ── helpers ─────────────────────────────────────────────────
def is_valid(tid: int, tokenizer) -> bool:
    txt = tokenizer.decode([tid], skip_special_tokens=False)
    if txt.strip() == "":
        return False
    if txt.startswith("<") and txt.endswith(">"):
        return False
    raw = tokenizer.convert_ids_to_tokens(tid)
    core = raw.lstrip("Ġ▁Ċ")
    if core == "":
        return False
    if all(unicodedata.category(c)[0] in {"P", "Z", "C"} for c in core):
        return False
    return True

def load_token_groups(path="token_groups.json"):
    with open(path) as f:
        groups=json.load(f)
    mp={}
    for g in groups:
        s=set(g)
        for t in s: mp[int(t)]=s
    return mp

def draw_bbox(img, box, color=(255,0,0), width=3):
    pil = Image.fromarray(img)
    draw=ImageDraw.Draw(pil)
    x1,y1,x2,y2 = map(int, box)
    for w in range(width):
        draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=color)
    return np.array(pil)

@torch.no_grad()
def token_info(model, tokenizer, img_tensor, prompt, gt_text, tok2grp):
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN+"\n"+prompt)
    conv.append_message(conv.roles[1], None)
    prompt_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).to(device)

    gt_ids = tokenizer(gt_text, return_tensors='pt').input_ids[0].to(device)
    out_list = []

    for i in range(1, len(gt_ids)):
        gt_tid = gt_ids[i].item()
        if not is_valid(gt_tid, tokenizer):
            continue

        inp = torch.cat([prompt_ids, gt_ids[:i].unsqueeze(0)], dim=1)
        out = model(input_ids=inp, images=img_tensor, use_cache=True)
        p   = F.softmax(out.logits[:, -1], dim=-1).squeeze(0)

        group = {t for t in tok2grp.get(gt_tid, {gt_tid})
                 if is_valid(t, tokenizer)}
        if not group:
            continue

        rank_val = 0
        for tid in p.argsort(descending=True).tolist():
            if not is_valid(tid, tokenizer):
                continue
            rank_val += 1
            if tid in group:
                break

        token_surface = tokenizer.decode([gt_tid], skip_special_tokens=False)
        token_len = len(normalize_chars_for_ngram(token_surface))
        out_list.append((rank_val, token_len, token_surface))

    return out_list

def load_image_tensor(img_path, mask_path, image_processor, box_color=(255,0,0), box_width=4):
    img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or img is None:
        raise FileNotFoundError(f"Cannot read {img_path} or {mask_path}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        x1, y1, x2, y2 = 0, 0, img.shape[1]-1, img.shape[0]-1
    else:
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for w in range(box_width):
        draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=box_color)
    tensor = image_processor.preprocess(pil, return_tensors='pt')['pixel_values']
    tensor = tensor.to(device, dtype=torch.half)
    return tensor

# ── main ───────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_groups", default="token_groups.json")
    ap.add_argument("--dataset", default="refcoco")
    ap.add_argument("--mono_counts", default=None)
    ap.add_argument("--bi_counts",   default=None)
    ap.add_argument("--tri_counts",  default=None)
    args = ap.parse_args()

    dataset = args.dataset.lower()
    image_dir = f'./datasets/images/train2014/'
    train_mask_root = f'datasets/sam_gt_mask/{dataset}'
    train_lmdb = f'datasets/lmdb/{dataset}/train.lmdb'

    tokenizer, model, processor, _ = load_pretrained_model(
        "vip_llava/vip-llava-7b",
        model_name="vip-llava-7b",
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device=device)
    model.eval()

    ds = RefcocoTextAnnotation(
        dataset=dataset,
        lmdb_dir=train_lmdb,
        image_dir=image_dir,
        mask_dir=train_mask_root)
    loader = DataLoader(ds, batch_size=1, sampler=DistributedSampler(ds, world, rank, shuffle=False))

    tok2grp = load_token_groups(args.token_groups)

    uni_cnt = load_counts_file(args.mono_counts, 1) if args.mono_counts else collections.Counter()
    H_letter = entropy_bits_from_monogram_counts(uni_cnt) if uni_cnt else FALLBACK_H

    # n-gram priors (bi/tri)
    bi_cnt  = load_counts_file(args.bi_counts, 2)   if args.bi_counts  else collections.Counter()
    tri_cnt = load_counts_file(args.tri_counts, 3)  if args.tri_counts else collections.Counter()
    ngram_model = NGramModel(uni_cnt, bi_cnt, tri_cnt)

    # {"click":int, "typing":int, "cost_click_typing":int,
    #  "typing_dep":float, "typing_indep":float,
    #  "info_ours_dep":float, "info_ours_indep":float,
    #  "typing_1gram":float, "typing_2gram":float, "typing_3gram":float,
    #  "our_1gram":float,    "our_2gram":float,    "our_3gram":float,
    #  "gt_text": str}
    local_costs = {K: {} for K in TOPK_SET}
    local_ranks = []
    
    for gt_sents, meta in tqdm(loader, disable=(rank!=0)):
        index    = int(meta["index"][0])
        img_path = os.path.join(image_dir, meta["img_name"][0])
        mask_path= meta["mask_dir"][0]
        img_tensor = load_image_tensor(img_path, mask_path, processor)

        prompt = "Describe the object in the red box."
        for sent_idx, sent in enumerate(gt_sents):
            text = sent[0] if isinstance(sent, tuple) else sent

            for rank_val, token_len, token_surface in token_info(
                    model, tokenizer, img_tensor, prompt,
                    text, tok2grp):
                local_ranks.append(rank_val)

                seg_letters = normalize_chars_for_ngram(token_surface)
                bits_1 = ngram_model.token_bits(seg_letters, order=1) if seg_letters else 0.0
                bits_2 = ngram_model.token_bits(seg_letters, order=2) if seg_letters else 0.0
                bits_3 = ngram_model.token_bits(seg_letters, order=3) if seg_letters else 0.0

                typing_dep_cost   = H_letter
                typing_indep_cost = token_len * H_letter

                for K in TOPK_SET:
                    seg_dict = local_costs[K].setdefault(index, {})
                    entry    = seg_dict.setdefault(
                        int(sent_idx),
                        {
                            "click":0, "typing":0, "cost_click_typing":0,
                            "typing_dep":0.0, "typing_indep":0.0,
                            "info_ours_dep":0.0, "info_ours_indep":0.0,
                            "typing_1gram":0.0, "typing_2gram":0.0, "typing_3gram":0.0,
                            "our_1gram":0.0,    "our_2gram":0.0,    "our_3gram":0.0,
                            "gt_text": text
                        }
                    )

                    hit = (rank_val <= K)
                    scan_bits = math.log2(K + 1)

                    if hit:
                        entry["click"] += 1
                        entry["cost_click_typing"] += 1
                    else:
                        entry["typing"] += token_len
                        entry["cost_click_typing"] += token_len

                    entry["typing_dep"]   += typing_dep_cost
                    entry["typing_indep"] += typing_indep_cost

                    if hit:
                        entry["info_ours_dep"]   += scan_bits
                        entry["info_ours_indep"] += scan_bits
                    else:
                        entry["info_ours_dep"]   += scan_bits + H_letter
                        entry["info_ours_indep"] += scan_bits + token_len * H_letter

                    entry["typing_1gram"] += bits_1
                    entry["typing_2gram"] += bits_2
                    entry["typing_3gram"] += bits_3
                    if hit:
                        entry["our_1gram"] += scan_bits
                        entry["our_2gram"] += scan_bits
                        entry["our_3gram"] += scan_bits
                    else:
                        entry["our_1gram"] += scan_bits + bits_1
                        entry["our_2gram"] += scan_bits + bits_2
                        entry["our_3gram"] += scan_bits + bits_3

    if dist.is_initialized():
        dist.barrier()

        if rank == 0:
            gathered_ranks = [None] * world
        else:
            gathered_ranks = None

        try:
            dist.gather_object(local_ranks, object_gather_list=gathered_ranks, dst=0)
        except Exception as e:
            if rank == 0:
                print(f"[rank0] gather_object(local_ranks) failed: {e}")
            raise

        if rank == 0:
            all_ranks = [r for sub in gathered_ranks for r in (sub or [])]
        else:
            all_ranks = None

        merged_costs = {K: {} for K in TOPK_SET}
        for K in TOPK_SET:
            if rank == 0:
                recv_list = [None] * world
            else:
                recv_list = None

            try:
                dist.gather_object(local_costs[K], object_gather_list=recv_list, dst=0)
            except Exception as e:
                if rank == 0:
                    print(f"[rank0] gather_object(costs K={K}) failed: {e}")
                raise

            if rank == 0:
                for d in recv_list:
                    for seg, sdict in d.items():
                        tgt_seg = merged_costs[K].setdefault(seg, {})
                        for sidx, vals in sdict.items():
                            tgt = tgt_seg.setdefault(
                                sidx,
                                {
                                    "click":0, "typing":0, "cost_click_typing":0,
                                    "typing_dep":0.0, "typing_indep":0.0,
                                    "info_ours_dep":0.0, "info_ours_indep":0.0,
                                    "typing_1gram":0.0, "typing_2gram":0.0, "typing_3gram":0.0,
                                    "our_1gram":0.0,    "our_2gram":0.0,    "our_3gram":0.0,
                                    "gt_text":""
                                }
                            )
                            for key in tgt.keys():
                                if key == "gt_text":
                                    if not tgt[key]:
                                        tgt[key] = vals[key]
                                else:
                                    tgt[key] += vals[key]
        if rank == 0:
            local_costs = merged_costs

        dist.barrier()
    else:
        all_ranks = local_ranks

    if rank == 0:
        arr = np.array(all_ranks, dtype=np.int64)

        totals = {}
        for K, seg_dict in local_costs.items():
            t = {
                "click":0, "typing":0, "cost_click_typing":0,
                "typing_dep":0.0, "typing_indep":0.0,
                "info_ours_dep":0.0, "info_ours_indep":0.0,
                "typing_1gram":0.0, "typing_2gram":0.0, "typing_3gram":0.0,
                "our_1gram":0.0,    "our_2gram":0.0,    "our_3gram":0.0
            }
            for seg in seg_dict.values():
                for sent in seg.values():
                    for k2 in t.keys():
                        t[k2] += sent[k2]
            totals[K] = t

        os.makedirs(f'datasets/al_cost/text3/{dataset}', exist_ok=True)
        out_file = f'datasets/al_cost/text3/{dataset}/rank_stats.txt'
        with open(out_file, "w") as f:
            f.write("=== ViP-LLaVA Rank Statistics ===\n")
            f.write(f"Total tokens     : {len(arr)}\n")
            f.write(f"Mean rank        : {arr.mean():.2f}\n")
            f.write(f"Median rank      : {np.median(arr):.0f}\n")
            f.write(f"25 / 75 pct      : "
                    f"{np.percentile(arr,25):.0f} / {np.percentile(arr,75):.0f}\n")
            f.write(f"Min / Max rank   : {arr.min()} / {arr.max()}\n")
            for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500, 1000]:
                f.write(f"≤{k:<4}: {(arr<=k).mean()*100:6.2f}%\n")

            f.write("\n=== Total annotation cost (legacy: click + typing) ===\n")
            for K in TOPK_SET:
                f.write(f"Top-{K:<3}: {totals[K]['cost_click_typing']}\n")

            f.write("\n=== Information-theoretic costs (totals; bits; monogram H) ===\n")
            for K in TOPK_SET:
                t = totals[K]
                f.write(f"[Top-{K:>2}] typing_dep={t['typing_dep']:.2f} | "
                        f"typing_indep={t['typing_indep']:.2f} | "
                        f"ours_dep={t['info_ours_dep']:.2f} | "
                        f"ours_indep={t['info_ours_indep']:.2f}\n")

            f.write("\n=== N-gram based costs (token-level; totals; bits) ===\n")
            for K in TOPK_SET:
                t = totals[K]
                f.write(f"[Top-{K:>2}] typing_1gram={t['typing_1gram']:.2f} | "
                        f"our_1gram={t['our_1gram']:.2f} | "
                        f"typing_2gram={t['typing_2gram']:.2f} | "
                        f"our_2gram={t['our_2gram']:.2f} | "
                        f"typing_3gram={t['typing_3gram']:.2f} | "
                        f"our_3gram={t['our_3gram']:.2f}\n")

            f.write(f"\n[Info] Monogram-based H_letter = {H_letter:.4f} bits/char "
                    f"(fallback used: {int(not bool(uni_cnt))})\n")

        print(f"[✓] Rank stats written to {out_file}")

        for K, cdict in local_costs.items():
            fname = f"datasets/al_cost/text3/{dataset}/text_cost_{K}.json"
            with open(fname, "w") as f:
                json.dump(cdict, f, indent=2)
            print(f"[✓] Cost dict (Top-{K}, info-theory + ngram) written to {fname}")

    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
