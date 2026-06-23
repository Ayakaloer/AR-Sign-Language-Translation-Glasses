"""
关键点序列预处理 - 双手版本。训练和推理共用。

输入:
  seq      : (T, 2, 21, 3)  原始关键点
  presence : (T, 2)         哪只手在场 (0/1)

归一化策略 (双手特殊点):
  - 不能各手独立归一, 否则会丢掉两手相对位置 (手语里很关键)
  - 用 "可见手" 中第一只的手腕做共享原点 + 共享尺度
  - 缺失的手保持 0 (后面会被 presence 标志位忽略)

镜像增强:
  - x 取反 + 同时交换 hand[0] 与 hand[1] + 交换两个 presence
    (单纯翻 x 会让左手出现在画面右侧、但仍被标为 hand[0], 制造假数据)

输出:
  (T, FEATURE_DIM)  =  (T, 128)
  布局: [hand0_xyz(63), hand1_xyz(63), presence0, presence1]
"""
from __future__ import annotations
import numpy as np

from config import FEATURE_DIM, NUM_HANDS, NUM_KEYPOINTS, NUM_COORDS

WRIST = 0
MIDDLE_MCP = 9
EPS = 1e-6


def _shared_anchor(seq: np.ndarray, presence: np.ndarray):
    """
    选 anchor 手: 整段中可见帧最多的那只手, 取整段的平均手腕 / 平均尺度。
    这样不会因为某一帧的抖动让所有点跳动。
    """
    visible_count = presence.sum(axis=0)   # (2,)
    if visible_count.max() < 1:
        return np.zeros(3, dtype=np.float32), 1.0

    anchor_hand = int(np.argmax(visible_count))
    mask = presence[:, anchor_hand] > 0.5
    wrist_pts = seq[mask, anchor_hand, WRIST, :]            # (Tv, 3)
    ref_pts = seq[mask, anchor_hand, MIDDLE_MCP, :] - wrist_pts
    origin = wrist_pts.mean(axis=0)
    scale = float(np.linalg.norm(ref_pts, axis=1).mean())
    if scale < EPS:
        scale = 1.0
    return origin.astype(np.float32), scale


def normalize_keypoints(seq: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """
    seq: (T, 2, 21, 3), presence: (T, 2)
    return 同形状归一化结果, 缺失的手保持 0
    """
    seq = seq.astype(np.float32, copy=True)
    origin, scale = _shared_anchor(seq, presence)

    # 平移 + 缩放仅作用于"在场"的手
    out = np.zeros_like(seq)
    for hi in range(NUM_HANDS):
        mask = presence[:, hi] > 0.5
        out[mask, hi] = (seq[mask, hi] - origin) / scale
    return out


def mirror_horizontal(seq: np.ndarray, presence: np.ndarray):
    """x 取反 + 交换两手, 模拟镜像视角。"""
    seq2 = seq.copy()
    seq2[..., 0] = -seq2[..., 0]
    seq2 = seq2[:, ::-1, :, :].copy()
    presence2 = presence[:, ::-1].copy()
    return seq2, presence2


def add_jitter(seq: np.ndarray, presence: np.ndarray, std: float = 0.01) -> np.ndarray:
    """只对在场的手加噪声, 缺失手保持 0。"""
    out = seq.copy()
    noise = np.random.normal(0, std, seq.shape).astype(seq.dtype)
    for hi in range(NUM_HANDS):
        mask = presence[:, hi] > 0.5
        out[mask, hi] = out[mask, hi] + noise[mask, hi]
    return out


def flatten(seq: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """
    (T, 2, 21, 3) + (T, 2)  ->  (T, 128)
    布局: hand0_xyz_63 | hand1_xyz_63 | pres0 | pres1
    """
    T = seq.shape[0]
    coords = seq.reshape(T, NUM_HANDS, NUM_KEYPOINTS * NUM_COORDS)   # (T, 2, 63)
    coords = coords.reshape(T, NUM_HANDS * NUM_KEYPOINTS * NUM_COORDS)  # (T, 126)
    out = np.concatenate([coords, presence.astype(np.float32)], axis=1)  # (T, 128)
    assert out.shape[1] == FEATURE_DIM, f"shape mismatch: {out.shape[1]} vs {FEATURE_DIM}"
    return out


def preprocess(seq: np.ndarray, presence: np.ndarray, augment: bool = False) -> np.ndarray:
    """完整流水线: 原始关键点 -> 模型输入 (T, 128)。"""
    if augment and np.random.rand() < 0.5:
        seq, presence = mirror_horizontal(seq, presence)
    seq = normalize_keypoints(seq, presence)
    if augment:
        seq = add_jitter(seq, presence, std=0.01)
    return flatten(seq, presence)
