"""
MediaPipe Hands 封装 - 双手版本。

输出固定形状的张量, 没检测到的手用 0 填充, 单独用 presence 数组标记是否存在。
按图像 x 坐标排序: 左侧的手放 hand[0], 右侧的手放 hand[1]。这样保证顺序稳定,
不依赖 MediaPipe 的 handedness 判断 (它在远距离时会抖动)。
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

NUM_HANDS = 2
NUM_KP = 21


class HandLandmarker:
    def __init__(
        self,
        max_num_hands: int = NUM_HANDS,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        返回:
          kp       : (2, 21, 3) float32  关键点, 缺失的手填 0
          presence : (2,) float32        0=缺失, 1=可见
        排序: 按手腕(关键点 0) 的 x 坐标升序, 左手在前。
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)

        kp = np.zeros((NUM_HANDS, NUM_KP, 3), dtype=np.float32)
        presence = np.zeros(NUM_HANDS, dtype=np.float32)

        if not res.multi_hand_landmarks:
            return kp, presence

        # 收集所有检测到的手 (最多 NUM_HANDS 只)
        hands_list: list[np.ndarray] = []
        for hand_lm in res.multi_hand_landmarks[:NUM_HANDS]:
            arr = np.empty((NUM_KP, 3), dtype=np.float32)
            for i, p in enumerate(hand_lm.landmark):
                arr[i, 0] = p.x
                arr[i, 1] = p.y
                arr[i, 2] = p.z
            hands_list.append(arr)

        # 按手腕 x 坐标排序 (画面左侧 -> hand[0])
        hands_list.sort(key=lambda h: h[0, 0])

        for i, h in enumerate(hands_list):
            kp[i] = h
            presence[i] = 1.0
        return kp, presence

    def draw(self, frame_bgr: np.ndarray, kp: np.ndarray, presence: np.ndarray) -> np.ndarray:
        h_img, w_img = frame_bgr.shape[:2]
        out = frame_bgr.copy()
        connections = self._mp_hands.HAND_CONNECTIONS
        colors_skel = [(0, 255, 0), (0, 200, 255)]   # 左手绿, 右手橙
        colors_pt = [(0, 0, 255), (255, 0, 0)]
        for hi in range(NUM_HANDS):
            if presence[hi] < 0.5:
                continue
            pts = [(int(kp[hi, i, 0] * w_img), int(kp[hi, i, 1] * h_img)) for i in range(NUM_KP)]
            for a, b in connections:
                cv2.line(out, pts[a], pts[b], colors_skel[hi], 2)
            for x, y in pts:
                cv2.circle(out, (x, y), 3, colors_pt[hi], -1)
        return out

    def close(self):
        self._hands.close()
