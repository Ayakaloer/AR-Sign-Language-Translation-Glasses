"""
语音播报: 把识别到的词读出来。

实现细节:
  - 用独立线程 + 队列, 不阻塞推理主循环
  - Windows: 用系统自带 SAPI (pyttsx3, 离线, 无需联网)
  - 同一个词短时间内不重复播 (跟 infer 的冷却互补)

依赖: pip install pyttsx3
"""
from __future__ import annotations
import queue
import threading
import time
from typing import Optional


class TTSWorker:
    def __init__(self, rate: int = 180, repeat_cooldown: float = 1.5):
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            raise RuntimeError("缺少 pyttsx3, 请: pip install pyttsx3") from None
        self._rate = rate
        self._cooldown = repeat_cooldown
        self._q: queue.Queue[Optional[str]] = queue.Queue(maxsize=8)
        self._last_text: str = ""
        self._last_time: float = 0.0
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def speak(self, text: str):
        """非阻塞: 把文字塞进队列, 由后台线程读出来。重复词会被丢弃。"""
        now = time.time()
        if text == self._last_text and now - self._last_time < self._cooldown:
            return
        self._last_text = text
        self._last_time = now
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass

    def stop(self):
        self._stopped.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def _loop(self):
        # 引擎只能在创建它的线程里使用, 所以放到这里
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        # 选第一个支持中文的语音
        for v in engine.getProperty("voices"):
            name = (v.name or "") + (v.id or "")
            if any(k in name.lower() for k in ("chinese", "zh", "huihui", "yaoyao", "kangkang")):
                engine.setProperty("voice", v.id)
                break

        while not self._stopped.is_set():
            text = self._q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[tts] {e}")
