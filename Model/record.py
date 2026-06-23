"""
数据采集脚本 (双手版本): 抓帧 -> MediaPipe 检测两只手 -> 录制带标签序列。

操作:
  - 选词: 数字键 0-9 (词表见 config.VOCAB)
  - 录制: 空格 开始/停止 一段, 录够 SEQ_LEN 帧自动保存
  - 退出: q

每个样本 npz:
  seq      : (T, 2, 21, 3) float32  双手关键点, 缺失手为 0
  presence : (T, 2)        float32  每帧每只手是否在场 (0/1)
  label    : int

只要 *至少有一只手* 在画面里就会写入这一帧。两只手都没检测到的帧会被跳过,
让你即使中途手晃出画面也不会污染数据。
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from config import VOCAB, SEQ_LEN, NUM_HANDS, NUM_KEYPOINTS, NUM_COORDS, DATA_DIR
from draw_cn import put_text_cn
from hands import HandLandmarker
from stream_reader import FrameReader


def parse_source(s: str) -> str | int:
    try:
        return int(s)
    except ValueError:
        return s


def count_existing(label: int) -> int:
    return len(list(DATA_DIR.glob(f"{label:02d}_*.npz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="本地摄像头 index (如 0) 或 P4 流 URL")
    args = ap.parse_args()

    source = parse_source(args.source)
    landmarker = HandLandmarker(max_num_hands=NUM_HANDS)

    current_label = 0
    recording = False
    seq_buf: list[np.ndarray] = []   # 每项 (2, 21, 3)
    pres_buf: list[np.ndarray] = []  # 每项 (2,)

    print("=" * 60)
    print("数据采集 (双手)")
    print("数字键 0-9: 切换当前词")
    print("空格: 开始/停止 录制 (录够 30 帧自动保存)")
    print("q: 退出")
    print("=" * 60)
    for i, w in enumerate(VOCAB):
        print(f"  [{i}] {w}  (已有 {count_existing(i)} 条)")

    with FrameReader(source) as reader:
        for frame in reader:
            kp, presence = landmarker.detect(frame)
            num_hands = int(presence.sum())
            vis = landmarker.draw(frame, kp, presence)

            # HUD
            txt_label = f"[{current_label}] {VOCAB[current_label]}  ({count_existing(current_label)} samples)"
            vis = put_text_cn(vis, txt_label, (10, 5), font_size=28, color_bgr=(0, 255, 255))
            status = f"REC {len(seq_buf)}/{SEQ_LEN}" if recording else "idle (SPACE to start)"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(vis, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            hand_txt = f"hands: {num_hands}/2"
            cv2.putText(vis, hand_txt, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if num_hands > 0 else (0, 0, 255), 2)

            cv2.imshow("record", vis)

            # 录制: 至少一只手在场才计入这一帧
            if recording and num_hands > 0:
                seq_buf.append(kp)
                pres_buf.append(presence)
                if len(seq_buf) >= SEQ_LEN:
                    seq = np.stack(seq_buf, axis=0)             # (T, 2, 21, 3)
                    pres = np.stack(pres_buf, axis=0)           # (T, 2)
                    fname = DATA_DIR / f"{current_label:02d}_{int(time.time()*1000)}.npz"
                    np.savez_compressed(fname, seq=seq, presence=pres, label=current_label)
                    print(f"saved {fname.name}  total={count_existing(current_label)}")
                    seq_buf.clear()
                    pres_buf.clear()
                    recording = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
                seq_buf.clear()
                pres_buf.clear()
                if recording:
                    print(f"开始录制: [{current_label}] {VOCAB[current_label]}")
            if ord("0") <= key <= ord("9"):
                idx = key - ord("0")
                if idx < len(VOCAB):
                    current_label = idx
                    recording = False
                    seq_buf.clear()
                    pres_buf.clear()
                    print(f"切换到: [{idx}] {VOCAB[idx]}")
            # 翻页: ] . n 下一词, [ , p 上一词
            if key in (ord("]"), ord("."), ord("n")):
                current_label = (current_label + 1) % len(VOCAB)
                recording = False
                seq_buf.clear()
                pres_buf.clear()
                print(f"==> [{current_label}] {VOCAB[current_label]}  ({count_existing(current_label)} samples)")
            if key in (ord("["), ord(","), ord("p")):
                current_label = (current_label - 1) % len(VOCAB)
                recording = False
                seq_buf.clear()
                pres_buf.clear()
                print(f"==> [{current_label}] {VOCAB[current_label]}  ({count_existing(current_label)} samples)")

    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
