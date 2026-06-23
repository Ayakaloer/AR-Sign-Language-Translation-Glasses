"""
在 OpenCV (BGR ndarray) 上绘制中文文字。

OpenCV 的 cv2.putText 不支持非 ASCII (中文会变 ?)。
做法: 把帧转成 PIL Image, 用 PIL 写文字, 再转回 BGR ndarray。
为了不每帧都重新加载字体, 把字体对象缓存起来。
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Windows 自带的字体, 几乎所有 Win10/11 都有
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑 Bold
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
]


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    # 兜底: PIL 默认字体, 但不一定支持中文
    return ImageFont.load_default()


def put_text_cn(
    img_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int = 24,
    color_bgr: tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """
    在 BGR 图上画中文。返回新图 (原图也会被修改, 但建议用返回值)。

    org: 文字左上角坐标 (x, y) -- 注意和 cv2.putText 的左下基线不同。
    color_bgr: BGR 顺序 (跟 OpenCV 一致)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, font=_font(font_size), fill=color_rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
