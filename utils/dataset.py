import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer


info = {
    'refcoco': {
        'train_gsam_object365': 156656,
        'train_gsam':           103230,
        'train':                42404,
        'val':                  3811,
        'val-test':             3811,
        'testA':                1975,
        'testB':                1810,
    },
    'refcoco+': {
        'train_gsam': 103216,
        'train':      42278,
        'val':        3805,
        'val-test':   3805,
        'testA':      1975,
        'testB':      1798,
    },
    'refcocog_u': {
        'train_gsam': 129162,
        'train':      42226,
        'val':        2573,
        'val-test':   2573,
        'test':       5023,
    },
}

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]
    sot = _tokenizer.encoder["<|startoftext|>"]
    eot = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot] + _tokenizer.encode(text) + [eot] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot
            else:
                raise RuntimeError(
                    f"Input {texts[i]!r} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


def loads_pyarrow(buf):
    return pa.deserialize(buf)


class _RefDatasetBase(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size, word_length):
        super().__init__()
        self.lmdb_dir   = lmdb_dir
        self.mask_dir   = mask_dir
        self.dataset    = dataset
        self.split      = split
        self.mode       = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,  0.40821073]).reshape(3, 1, 1)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys   = loads_pyarrow(txn.get(b'__keys__'))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return self.length

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

    def convert(self, img, mask=None):
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask


class RefDatasetCorrection(_RefDatasetBase):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length, exp_mode=False, use_first_only=False):
        super().__init__(lmdb_dir, mask_dir, dataset, split, mode, input_size, word_length)
        self.exp_mode        = exp_mode
        self.use_first_only  = use_first_only
        self.fixed_sent_idx  = None

    def set_sent_idx(self, idx):
        self.fixed_sent_idx = idx

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)

        ori_img  = cv2.imdecode(np.frombuffer(ref['img'], np.uint8), cv2.IMREAD_COLOR)
        img_name = ref['img_name']
        img      = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]

        seg_id   = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')

        sents     = ref['sents']
        num_sents = len(sents)
        mat, _   = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(img, mat, self.input_size, flags=cv2.INTER_CUBIC,
                             borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.warpAffine(mask, mat, self.input_size,
                              flags=cv2.INTER_LINEAR, borderValue=0.)
        mask = mask / 255.

        if self.use_first_only:
            sent = sents[0] if num_sents else " "
        elif self.fixed_sent_idx is not None:
            sent = sents[self.fixed_sent_idx % num_sents] if num_sents else " "
        else:
            sent = sents[int(np.random.choice(num_sents))] if num_sents else " "

        word_vec = tokenize(sent, self.word_length, True).squeeze(0)
        img, mask = self.convert(img, mask)
        if self.exp_mode:
            return {'img_name': img_name, 'mask_dir': mask_dir}
        return img, word_vec, mask, index, img_name


class RefDataset(_RefDatasetBase):
    def __getitem__(self, index):
        if self.env is None:
            self._init_db()
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)

        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8), cv2.IMREAD_COLOR)
        img_id  = ref['img_name']
        img     = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]

        seg_id   = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')

        idx   = np.random.choice(ref['num_sents'])
        sents = ref['sents']

        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(img, mat, self.input_size, flags=cv2.INTER_CUBIC,
                             borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

        if self.mode == 'train':
            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask, mat, self.input_size,
                                  flags=cv2.INTER_LINEAR, borderValue=0.)
            mask = mask / 255.
            sent = sents[idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img, mask = self.convert(img, mask)
            return img, word_vec, mask, index

        if self.mode == 'val':
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            params = {
                'mask_dir': mask_dir,
                'inverse' : mat_inv,
                'ori_size': np.array(img_size),
            }
            return img, word_vec, params

        if self.mode == 'test':
            img = self.convert(img)[0]
            params = {
                'ori_img' : ori_img,
                'seg_id'  : seg_id,
                'mask_dir': mask_dir,
                'inverse' : mat_inv,
                'ori_size': np.array(img_size),
                'sents'   : sents,
            }
            return img, params

        mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.warpAffine(mask, mat, self.input_size,
                              flags=cv2.INTER_LINEAR, borderValue=0.)
        mask = mask / 255.
        sent = sents[0]
        word_vec = tokenize(sent, self.word_length, True).squeeze(0)
        img, mask = self.convert(img, mask)
        return img, word_vec, mask, index, seg_id
