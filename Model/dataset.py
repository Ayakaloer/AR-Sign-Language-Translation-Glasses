"""
PyTorch Dataset (双手版本): npz -> 模型输入张量。

每个 npz:
  seq      : (T, 2, 21, 3) float32
  presence : (T, 2)        float32
  label    : int
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import preprocess
from config import DATA_DIR, SEQ_LEN, NUM_HANDS


class HandSignDataset(Dataset):
    def __init__(self, files: list[Path], augment: bool = False):
        self.files = files
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        seq = d["seq"]          # (T, 2, 21, 3)
        pres = d["presence"]    # (T, 2)
        label = int(d["label"])

        # 帧数兜底
        if seq.shape[0] != SEQ_LEN:
            seq = _pad_or_crop(seq, SEQ_LEN)
            pres = _pad_or_crop(pres, SEQ_LEN)

        x = preprocess(seq, pres, augment=self.augment)   # (T, 128)
        # Conv1d 要 channel-first: (128, T)
        x = x.transpose(1, 0).astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def _pad_or_crop(arr: np.ndarray, T: int) -> np.ndarray:
    if arr.shape[0] == T:
        return arr
    if arr.shape[0] > T:
        return arr[:T]
    pad = np.repeat(arr[-1:], T - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def list_all_samples(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(data_dir.glob("*.npz"))


def split_by_label(files: list[Path], val_ratio: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Path]] = {}
    for f in files:
        lbl = int(f.name.split("_", 1)[0])
        by_label.setdefault(lbl, []).append(f)
    train, val = [], []
    for lbl, fs in by_label.items():
        idx = rng.permutation(len(fs))
        n_val = max(1, int(len(fs) * val_ratio))
        val.extend(fs[i] for i in idx[:n_val])
        train.extend(fs[i] for i in idx[n_val:])
    return train, val
