"""
从 MJPEG 流或本地摄像头抓帧。

用法:
    with FrameReader("http://47.108.195.156:8081/stream") as r:
        for frame in r:
            ...

    with FrameReader(0) as r:   # 本地摄像头
        for frame in r:
            ...

HTTP 流: 后台线程只负责从网络读原始 JPEG 字节，
主线程在 __iter__ 里按需 cv2.imdecode，彻底解耦 IO 和 CPU。
"""
from __future__ import annotations
import threading
import time
from typing import Iterator

import cv2
import numpy as np
import requests


class FrameReader:
    """低延迟帧读取器。后台抓 JPEG，主线程解码。"""

    def __init__(self, source: str | int, timeout: float = 10.0):
        self.source = source
        self.timeout = timeout
        self._cap: cv2.VideoCapture | None = None
        self._raw: bytes | None = None       # HTTP: 最新 JPEG 字节
        self._frame: np.ndarray | None = None  # 本地摄像头: 解码后的帧
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._is_local = isinstance(source, int)

    def __enter__(self) -> "FrameReader":
        if self._is_local:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开摄像头: {self.source}")
            self._running = True
            self._thread = threading.Thread(target=self._grab_local, daemon=True)
        else:
            self._running = True
            self._thread = threading.Thread(target=self._grab_http, daemon=True)
        self._thread.start()
        self._wait_first()
        return self

    def _wait_first(self):
        t0 = time.time()
        while self._running:
            if self._is_local:
                if self._frame is not None:
                    return
            else:
                if self._raw is not None:
                    return
            if time.time() - t0 > self.timeout:
                raise RuntimeError(f"超时: {self.timeout}s 内没收到帧")
            time.sleep(0.01)

    def pause(self):
        """停止抓帧线程, 释放网络连接 (用于语音模式省带宽)"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def resume(self):
        """重新启动抓帧 (从语音模式返回)"""
        if self._is_local:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开摄像头: {self.source}")
            self._running = True
            self._thread = threading.Thread(target=self._grab_local, daemon=True)
        else:
            self._frame = None if not hasattr(self, '_raw') else self._frame
            self._raw = None
            self._running = True
            self._thread = threading.Thread(target=self._grab_http, daemon=True)
        self._thread.start()

    def __exit__(self, exc_type, exc, tb):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        while self._running:
            if self._is_local:
                with self._lock:
                    frame = self._frame
                if frame is None:
                    time.sleep(0.001)
                    continue
                yield frame
            else:
                with self._lock:
                    raw = self._raw
                if raw is None:
                    time.sleep(0.001)
                    continue
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img
                else:
                    time.sleep(0.001)

    def _grab_local(self):
        assert self._cap is not None
        while self._running:
            ok = self._cap.grab()
            if not ok:
                time.sleep(0.01)
                continue
            ok, frame = self._cap.retrieve()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame

    def _grab_http(self):
        """后台线程只做网络 IO: 读 chunk → 找边界 → 存原始 JPEG 字节。不解码。"""
        import socket
        session = requests.Session()
        CHUNK = 65536
        MAX_BUF = 1 * 1024 * 1024

        while self._running:
            try:
                resp = session.get(self.source, stream=True, timeout=(3, 6))
                if resp.status_code != 200:
                    time.sleep(0.5)
                    continue

                sock = resp.raw._fp.fp.raw._sock
                if sock:
                    sock.settimeout(2.0)

                content_type = resp.headers.get("Content-Type", "")
                boundary_bytes = None
                if "multipart" in content_type:
                    for part in content_type.split(";"):
                        part = part.strip()
                        if part.lower().startswith("boundary="):
                            boundary_bytes = part.split("=", 1)[1].strip().encode()
                            break
                    if not boundary_bytes:
                        time.sleep(0.5)
                        continue

                buf = b""
                soi_marker = b"\xff\xd8"
                eoi_marker = b"\xff\xd9"

                while self._running:
                    try:
                        chunk = resp.raw.read(CHUNK)
                    except Exception:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) > MAX_BUF:
                        buf = buf[-MAX_BUF // 2:]

                    while True:
                        if boundary_bytes:
                            idx = buf.find(boundary_bytes)
                            if idx < 0:
                                break
                            start = idx + len(boundary_bytes)
                            rest = buf[start:]
                            next_idx = rest.find(boundary_bytes)
                            if next_idx < 0:
                                break
                            section = rest[:next_idx]
                            buf = rest[next_idx:]
                            hdr = section.find(b"\r\n\r\n")
                            jpeg = section[hdr + 4:] if hdr >= 0 else b""
                        else:
                            soi = buf.find(soi_marker)
                            if soi < 0:
                                buf = buf[-3:]
                                break
                            eoi = buf.find(eoi_marker, soi)
                            if eoi < 0:
                                break
                            jpeg = buf[soi:eoi + 2]
                            buf = buf[eoi + 2:]

                        if jpeg:
                            with self._lock:
                                self._raw = jpeg

            except (requests.RequestException, socket.timeout):
                time.sleep(0.3)
        session.close()
