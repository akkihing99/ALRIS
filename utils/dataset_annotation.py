import sys
sys.path.append('./')
import os
from typing import List, Union
# from PIL import Image
# import io
import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pandas as pd
from pycocotools.coco import COCO
import random

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    }
}

class RefcocoMaskAnnotation(Dataset):
    def __init__(self, dataset, lmdb_dir, image_dir, gt_mask_dir):
        self.image_dir = image_dir
        self.gt_mask_dir = gt_mask_dir
        self.lmdb_dir = lmdb_dir
        self.env = None
        self.length = info[dataset]['train']

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                                subdir=os.path.isdir(self.lmdb_dir),
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))
            
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        gt_sents = ref['sents']
        img_name = ref['img_name']
        seg_id = ref['seg_id']
        sent_ids = ref.get('sent_id', None)

        img_dir = os.path.join(self.image_dir, img_name)

        mask_dir = os.path.join(self.gt_mask_dir, f'{seg_id}.png')

        meta = {
            'img_name': img_name,
            'seg_id': seg_id,
            'mask_dir': mask_dir,
            'index': index,
            'sents': gt_sents,
            'sent_ids': sent_ids
        }

        return meta
    

    def transform(self, img, mask=None):
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None
    


class RefcocoTextAnnotation(Dataset):
    def __init__(self, dataset, lmdb_dir, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.lmdb_dir = lmdb_dir
        self.env = None
        self.length = info[dataset]['train']

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                                subdir=os.path.isdir(self.lmdb_dir),
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))
            
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        gt_sents = ref['sents']
        img_name = ref['img_name']
        seg_id = ref['seg_id']
        
        img_dir = os.path.join(self.image_dir, img_name)

        mask_dir = os.path.join(self.mask_dir, f'{seg_id}.png')

        meta = {
            'img_name': img_name,
            'seg_id': seg_id,
            'mask_dir': mask_dir,
            'index': index,
        }

        return gt_sents, meta
    

    def transform(self, img, mask=None):
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

class RefcocoRIStoREC(Dataset):
    def __init__(self, csv_dir, image_dir, mask_dir, input_size, word_length, doc_mode=False,
                 doc_filter='value', seg_model='sam', beam_search=False, corretness_mode=False,
                 uniqueness=False, correctness=False, exp_mode=False, only_top_doc=False, top_k=5):
        self.csv_dir = csv_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.doc_mode = doc_mode
        self.doc_filter = doc_filter
        self.seg_model = seg_model
        self.beam_search = beam_search
        self.df = pd.read_csv(self.csv_dir)
        self.uniqueness = uniqueness
        self.correctness = correctness
        self.exp_mode = exp_mode
        self.only_top_doc = only_top_doc
        self.top_k = top_k

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_name = row['img_name']
        seg_id = row['seg_id']
        num_sent = row['num_sent']
        pseudo_caption = eval(row['pseudo_caption'])

        beam_keep_idx = None
        if self.beam_search:
            beam_keep_idx = [i for i in range(len(pseudo_caption)) if i % 11 == 1]
            pseudo_caption = [pseudo_caption[i] for i in beam_keep_idx]

        img_dir = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_size = img.shape[:2]
        mat, mat_inv = self.getTransformMat(img_size, inverse=True)

        img = cv2.warpAffine(
            img, mat, self.input_size, flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        )

        mask_dir = os.path.join(self.mask_dir, f'{seg_id}.png')

        if self.doc_mode:
            doc = eval(row['doc'])

            if beam_keep_idx is not None:
                doc = [doc[i] for i in beam_keep_idx]

            L = min(len(pseudo_caption), len(doc))
            pseudo_caption = pseudo_caption[:L]
            doc = doc[:L]

            K = 1 if self.only_top_doc else max(1, min(self.top_k, L))

            sorted_idx = np.argsort(doc)[::-1][:K]
            selected_indices = list(sorted_idx)

        else:
            L = len(pseudo_caption)
            K = max(1, min(self.top_k, L))
            selected_indices = np.random.choice(L, size=K, replace=False).tolist()

        selected_captions = [pseudo_caption[i] for i in selected_indices]

        mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
        mask = cv2.warpAffine(mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.)
        mask = mask / 255
        img, mask = self.transform(img, mask)

        mask_dir = os.path.join(self.mask_dir, f'{seg_id}.png')

        meta = {
            'img_name': img_name,
            'seg_id': seg_id,
            'mask_dir': mask_dir,
            'index': index,
            'sents': selected_captions
        }

        return meta
    
    def transform(self, img, mask=None):
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

class RefcocoPseudoRISAnnotation(Dataset):
    def __init__(self, csv_path, image_dir, mask_dir):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.df = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        img_name = row['img_name']
        img_id = str(row['img_id'])
        seg_id = str(row['seg_id'])
        
        pseudo_caption = eval(row['pseudo_caption'])
        
        try:
            doc = eval(row['doc'])
            best_idx = int(np.argmax(doc))
            best_text = pseudo_caption[best_idx]
            best_doc_score = doc[best_idx]
        except (KeyError, ValueError, SyntaxError):
            best_text = pseudo_caption[0]
            best_doc_score = 0.0
        
        # Mask path: datasets/cutler_sam_refcoco/img_id/seg_id.png
        mask_path = os.path.join(self.mask_dir, img_id, f'{seg_id}.png')
        
        pseudo_key = f"{img_id}_{seg_id}"
        
        meta = {
            'img_name': img_name,
            'img_id': img_id,
            'seg_id': seg_id,
            'pseudo_key': pseudo_key,
            'mask_dir': mask_path,
            'index': index,
            'best_text': best_text,
            'best_doc_score': best_doc_score,
            'all_texts': pseudo_caption,
        }
        
        return meta
    
if __name__ == '__main__':

    dataset = 'refcoco+'
    seg_model = 'cutler_sam'
    model_type = 'cc3m'
    csv_dir = f'./pseudo_supervision/{seg_model}/base_{model_type}.csv'
    image_dir = f'./datasets/images/train2014/'
    gt_mask_dir = f'./datasets/pseudo_masks/{seg_model}'
    train_lmdb =f'datasets/lmdb/{dataset}/train.lmdb'
    doc_mode = True

    dataset = RefcocoTextAnnotation(lmdb_dir=train_lmdb,
                                    csv_dir=csv_dir,
                                    image_dir=image_dir,
                                    gt_mask_dir=gt_mask_dir,
                                    input_size=416,
                                    word_length=17)

    a = dataset[0]


