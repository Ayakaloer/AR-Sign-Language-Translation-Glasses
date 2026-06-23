"""
实时手语识别: 待机(OLED显示时间+天气) → "请"唤醒 → 识别模式(OLED实时显示翻译)。

使用:
    python infer.py                                # 本地摄像头
    python infer.py --source http://<P4-IP>:81/stream

按键:
    q  退出
    space  暂停/继续
    m  手动切换 待机/识别 模式
"""
from __future__ import annotations
import argparse
import collections
import sys
import threading
import time
from datetime import datetime, timedelta

# Windows 终端 UTF-8 编码，解决中文乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import requests
import torch

from config import (CKPT_DIR, NUM_HANDS, NUM_KEYPOINTS, NUM_COORDS,
                    SEQ_LEN, BUF_LEN, VOCAB, WEATHER_CITY)
from draw_cn import put_text_cn
from hands import HandLandmarker
from model import SignNet
from preprocess import preprocess
from stream_reader import FrameReader
from tts import TTSWorker


# ----- 参数 -----
CONF_THRESH = 0.85
STABLE_N = 8
COOLDOWN_S = 1.5
SENTENCE_TIMEOUT = 3.5
MOTION_THRESH = 0.05
DETECT_EVERY_N = 5     # 每 5 帧才跑一次 MediaPipe, 降低延迟
MAX_FRAME_W = 400       # 缩放帧宽加速检测, 不过小以免误检

P4_API_URL = "http://10.244.3.204:80/api/control"
MIN_P4_INTERVAL = 5.0    # 两次 P4 推送最小间隔 (秒)


# P4 局域网设备, 不走系统代理
_p4_session = requests.Session()
_p4_session.trust_env = False
_last_p4_push = 0.0

def _p4_send(text: str, timeout: float = 2.0):
    def _send():
        try:
            hex_str = to_gbk_hex(text)
            resp = _p4_session.post(P4_API_URL, json={"hex": hex_str}, timeout=timeout)
            print(f"[p4] {resp.text}")
        except Exception as e:
            print(f"[warn] P4 push failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

def push_to_p4(text: str, timeout: float = 2.0):
    global _last_p4_push
    now = time.time()
    if now - _last_p4_push < MIN_P4_INTERVAL:
        return
    _last_p4_push = now
    _p4_send(text, timeout)

def push_to_p4_urgent(text: str, timeout: float = 2.0):
    """绕过节流, 立即发送 (用于模式切换等关键消息)"""
    _p4_send(text, timeout)

def push_raw_msg(msg: str, timeout: float = 2.0):
    """发送原始 message 格式 (非 hex), 用于特殊指令如 '985'"""
    def _send():
        try:
            resp = _p4_session.post(P4_API_URL, json={"message": msg}, timeout=timeout)
            print(f"[p4] {resp.text}")
        except Exception as e:
            print(f"[warn] P4 push failed: {e}")
    threading.Thread(target=_send, daemon=True).start()


def get_network_time() -> datetime | None:
    try:
        resp = requests.get("http://worldtimeapi.org/api/timezone/Asia/Shanghai", timeout=3)
        if resp.status_code == 200:
            dt_str = resp.json().get("datetime", "")
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    return None


def count_fingers(kp: np.ndarray) -> int:
    """从 MediaPipe 手部关键点计算伸出指头数 (0-5)。kp: (21, 3)
    用指尖到手腕距离 vs PIP到手腕距离判断，旋转不敏感。"""
    wrist = kp[0]
    tips  = [4,  8,  12, 16, 20]
    pips  = [3,  6,  10, 14, 18]
    n = 0
    for t, p in zip(tips, pips):
        d_tip = np.linalg.norm(kp[t] - wrist)
        d_pip = np.linalg.norm(kp[p] - wrist)
        if d_tip > d_pip * 1.2:  # 指尖比关节远 20% = 手指伸出
            n += 1
    return n


def fetch_weather(city: str) -> str:
    try:
        # 中文天气描述 + 温度
        resp = requests.get(f"http://wttr.in/{city}?format=%t", timeout=3)
        text = resp.text.strip()
        # OLED 字库无 ° 符号, 替换为空格或省略
        text = text.replace("°", "")
        return text
    except Exception:
        return "--"

def cleanup_sentence(words: list[str]) -> list[str]:
    """柔性清理: 仅合并 3 次以上连续重复 (如'我我我'→'我'), 保留两次重复。"""
    if len(words) <= 2:
        return words
    result = []
    streak = 1
    for i, w in enumerate(words):
        if i > 0 and w == words[i - 1]:
            streak += 1
        else:
            streak = 1
        if streak <= 2:  # 最多保留连续 2 个
            result.append(w)
    return result


def to_gbk_hex(text: str) -> str:
    """编码为 GBK 十六进制，滤掉标点、emoji 和 GBK 不支持的字符。"""
    import re
    # 保留: 中文、字母、数字、空格、/、:、+、°、-
    clean = re.sub(r'[^一-鿿\w\s/:\+\-°]', '', text)
    clean = clean.encode("gbk", errors="replace").decode("gbk", errors="replace")
    return clean.encode("gbk").hex()


def reduce_camera_quality(quality: int = 15):
    """降低 P4 摄像头 JPEG 质量。失败静默不阻塞。"""
    try:
        api = P4_API_URL.replace("/api/control", "/api/set_camera_config")
        requests.post(api, json={"index": 0, "image_format": 0, "jpeg_quality": quality}, timeout=2)
    except Exception:
        pass  # 不支持此 API 的 P4 静默跳过


def parse_source(s: str) -> str | int:
    try:
        return int(s)
    except ValueError:
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="本地摄像头 index 或 P4 流 URL")
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--no-tts", action="store_true", help="关闭语音播报")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SignNet(num_classes=len(VOCAB)).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt}  (epoch {ckpt.get('epoch', '?')}, val_acc {ckpt.get('val_acc', '?')})")

    landmarker = HandLandmarker(max_num_hands=NUM_HANDS)

    tts: TTSWorker | None = None
    if not args.no_tts:
        try:
            tts = TTSWorker()
            print("tts: 已启用")
        except Exception as e:
            print(f"tts: 启动失败 ({e})")

    # ----- 状态机 -----
    mode: str = "idle"          # "idle" | "recognize" | "exit_prompt"
    FINGER_STABLE_N = 5         # 手势稳定帧数
    IDLE_OLED_INTERVAL = 120.0
    last_oled_push = 0.0
    last_pushed_minute = -1

    weather_str = fetch_weather(WEATHER_CITY)
    last_weather_fetch = time.time()
    network_time: datetime | None = get_network_time()
    last_ntp_fetch = time.time()

    # 缓冲区
    kp_buf: collections.deque[np.ndarray] = collections.deque(maxlen=BUF_LEN)
    pres_buf: collections.deque[np.ndarray] = collections.deque(maxlen=BUF_LEN)

    last_label = -1
    last_label_time = 0.0
    stable_label = -1
    stable_count = 0
    paused = False

    sentence_words: list[str] = []
    last_hand_time = time.time()
    ref_wrist: np.ndarray | None = None

    fps_t = time.time()
    fps_n = 0
    fps = 0.0
    frame_count = 0
    kp = np.zeros((NUM_HANDS, NUM_KEYPOINTS, NUM_COORDS), dtype=np.float32)
    presence = np.zeros(NUM_HANDS, dtype=np.float32)
    num_hands = 0

    finger_stable_val = -1
    finger_stable_count = 0
    last_no_hand_time = time.time()
    wake_blocked_until = 0.0  # 退出后冷却, 防止残留手势误触发

    print("待机 — 比1进入识别 | m键切换")

    source = parse_source(args.source)
    with FrameReader(source) as reader:
        for frame in reader:
            frame_count += 1
            do_detect = (frame_count % DETECT_EVERY_N == 0)
            if paused:
                cv2.imshow("infer", frame)
                if (cv2.waitKey(30) & 0xFF) == ord(" "):
                    paused = False
                continue

            # --- 手部检测 ---
            if do_detect or frame_count == 1:
                # 缩小帧加速 MediaPipe, keypoints 坐标归一化不受影响
                if MAX_FRAME_W and frame.shape[1] > MAX_FRAME_W:
                    scale = MAX_FRAME_W / frame.shape[1]
                    small = cv2.resize(frame, (MAX_FRAME_W, int(frame.shape[0] * scale)))
                else:
                    small = frame
                kp, presence = landmarker.detect(small)
                num_hands = int(presence.sum())
                if num_hands > 0:
                    last_hand_time = time.time()
                    if ref_wrist is None:
                        ref_wrist = kp[0, 0, :2].copy()
                else:
                    ref_wrist = None

            kp_buf.append(kp)
            pres_buf.append(presence)

            label = -1
            conf = 0.0
            if len(kp_buf) == BUF_LEN:
                pres_arr = np.stack(list(pres_buf), axis=0)
                if pres_arr.sum() > 0:
                    seq = np.stack(list(kp_buf), axis=0)
                    indices = np.linspace(0, BUF_LEN - 1, SEQ_LEN, dtype=int)
                    seq = seq[indices]
                    pres_arr = pres_arr[indices]
                    x = preprocess(seq, pres_arr, augment=False)
                    x = x.transpose(1, 0).astype(np.float32, copy=False)
                    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x_t)
                        prob = torch.softmax(logits, dim=1)[0]
                        idx = int(prob.argmax().item())
                        c = float(prob[idx].item())

                    if c >= CONF_THRESH and num_hands > 0 and ref_wrist is not None:
                        cur_wrist = kp[0, 0, :2]
                        if np.linalg.norm(cur_wrist - ref_wrist) < MOTION_THRESH:
                            c = 0.0

                    if c >= CONF_THRESH:
                        label = idx
                        conf = c

            now = time.time()

            # ====================================================
            #  待机模式
            # ====================================================
            if mode == "idle":
                # 获取当前时间
                if now - last_ntp_fetch > 120:
                    nt = get_network_time()
                    if nt:
                        network_time = nt
                    last_ntp_fetch = now
                if network_time:
                    elapsed = now - last_ntp_fetch
                    cur_dt = network_time + timedelta(seconds=elapsed)
                else:
                    cur_dt = datetime.now()
                cur_minute = cur_dt.minute

                # 当分钟变化或超过推送间隔时, 推送时间天气到 OLED
                if cur_minute != last_pushed_minute or now - last_oled_push > IDLE_OLED_INTERVAL:
                    push_to_p4(f"{WEATHER_CITY} {weather_str}\n{cur_dt.strftime('%m/%d %H:%M')}\n比1进入手语识别\n比2进入语音识别")
                    last_oled_push = now
                    last_pushed_minute = cur_minute

                if now - last_weather_fetch > 600:
                    weather_str = fetch_weather(WEATHER_CITY)
                    last_weather_fetch = now

                # 检测手指数量: 1=手语识别, 2=语音识别 (冷却期跳过)
                cur_fingers = count_fingers(kp[0]) if num_hands > 0 else 0
                finger_action = None
                if now < wake_blocked_until:
                    cur_fingers = 0
                elif cur_fingers in (1, 2):
                    if finger_stable_val == cur_fingers:
                        finger_stable_count += 1
                    else:
                        finger_stable_val = cur_fingers
                        finger_stable_count = 1
                    if finger_stable_count >= FINGER_STABLE_N:
                        finger_action = cur_fingers
                else:
                    finger_stable_val = -1
                    finger_stable_count = 0

                if finger_action == 1:
                    mode = "recognize"
                    push_to_p4("进入手语识别")
                    print(f"[{time.strftime('%H:%M:%S')}] >>> 进入手语识别")
                    if tts:
                        tts.speak("开始识别")
                    kp_buf.clear()
                    pres_buf.clear()
                    sentence_words.clear()
                    last_label = -1
                    last_no_hand_time = now
                elif finger_action == 2:
                    mode = "voice"
                    push_raw_msg("985")
                    reduce_camera_quality(15)   # 降低画质释放 P4 带宽
                    print(f"[{time.strftime('%H:%M:%S')}] >>> 进入语音识别")

                # 待机画面
                vis = frame
                cv2.putText(vis, f"hand:{num_hands} finger:{cur_fingers}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
                if finger_action:
                    hint = "手语" if finger_action == 1 else "语音"
                    vis = put_text_cn(vis, f">>> 进入{hint}识别 <<<", (20, 55), font_size=28, color_bgr=(0, 255, 255))
                cv2.imshow("infer", vis)

            # ====================================================
            #  语音识别模式 (极轻量: 仅隔帧检测3手势, 不做推理)
            # ====================================================
            elif mode == "voice":
                if frame_count % 15 == 0:  # 每 ~0.5s 检测一次
                    kp_v, presence_v = landmarker.detect(frame)
                    num_hands_v = int(presence_v.sum())
                    cur_f = count_fingers(kp_v[0]) if num_hands_v > 0 else 0
                    if cur_f == 3:
                        if finger_stable_val == 3:
                            finger_stable_count += 1
                        else:
                            finger_stable_val = 3
                            finger_stable_count = 1
                        if finger_stable_count >= 3:  # 快速响应
                            mode = "idle"
                            wake_blocked_until = now + 3.0
                            push_raw_msg("211")
                            reduce_camera_quality(30)  # 恢复画质
                            if network_time:
                                cur_dt = network_time + timedelta(seconds=now - last_ntp_fetch)
                            else:
                                cur_dt = datetime.now()
                            push_to_p4_urgent(f"{WEATHER_CITY} {weather_str}\n{cur_dt.strftime('%m/%d %H:%M')}\n比1进入手语识别\n比2进入语音识别")
                            print(f"[{time.strftime('%H:%M:%S')}] <<< 退出语音识别")
                            continue
                    else:
                        finger_stable_val = -1
                        finger_stable_count = 0

                if frame_count % 30 == 0:
                    cv2.imshow("infer", frame)

            # ====================================================
            #  手语识别模式 / 退出确认模式
            # ====================================================
            elif mode != "idle":
                EXIT_PROMPT_TIMEOUT = 7.0  # 无手 N 秒后弹出退出确认

                if mode == "recognize":
                    # 7秒无手 → 弹出退出确认
                    if num_hands > 0:
                        last_no_hand_time = now
                    elif now - last_no_hand_time > EXIT_PROMPT_TIMEOUT:
                        mode = "exit_prompt"
                        push_to_p4("是否退出手语识别 2继续 3退出")
                        print(f"[{time.strftime('%H:%M:%S')}] ??? 退出确认")
                        finger_stable_count = 0
                        finger_stable_val = -1

                elif mode == "exit_prompt":
                    # 检测 2(继续) 或 3(退出) 手势
                    cur_f = count_fingers(kp[0]) if num_hands > 0 else -1
                    action = None
                    if cur_f == 2:
                        if finger_stable_val == 2:
                            finger_stable_count += 1
                        else:
                            finger_stable_val = 2
                            finger_stable_count = 1
                        if finger_stable_count >= FINGER_STABLE_N:
                            action = "stay"
                    elif cur_f == 3:
                        if finger_stable_val == 3:
                            finger_stable_count += 1
                        else:
                            finger_stable_val = 3
                            finger_stable_count = 1
                        if finger_stable_count >= FINGER_STABLE_N:
                            action = "exit"
                    else:
                        finger_stable_val = -1
                        finger_stable_count = 0

                    if action == "exit":
                        mode = "idle"
                        wake_blocked_until = now + 3.0
                        push_to_p4("已退出手语识别")
                        if network_time:
                            cur_dt = network_time + timedelta(seconds=now - last_ntp_fetch)
                        else:
                            cur_dt = datetime.now()
                        push_to_p4_urgent(f"{WEATHER_CITY} {weather_str}\n{cur_dt.strftime('%m/%d %H:%M')}\n比1进入手语识别\n比2进入语音识别")
                        print(f"[{time.strftime('%H:%M:%S')}] <<< 退出识别 (3)")
                        sentence_words.clear()
                        kp_buf.clear()
                        pres_buf.clear()
                        continue
                    elif action == "stay":
                        mode = "recognize"
                        last_no_hand_time = now
                        print(f"[{time.strftime('%H:%M:%S')}] ... 继续识别 (2)")
                        continue

                    # 退出确认超时 (再过10秒无反应自动退出)
                    if now - last_no_hand_time > EXIT_PROMPT_TIMEOUT + 10:
                        mode = "idle"
                        wake_blocked_until = now + 3.0
                        push_to_p4("已退出手语识别")
                        if network_time:
                            cur_dt = network_time + timedelta(seconds=now - last_ntp_fetch)
                        else:
                            cur_dt = datetime.now()
                        push_to_p4_urgent(f"{WEATHER_CITY} {weather_str}\n{cur_dt.strftime('%m/%d %H:%M')}\n比1进入手语识别\n比2进入语音识别")
                        print(f"[{time.strftime('%H:%M:%S')}] <<< 退出识别 (超时)")
                        sentence_words.clear()
                        kp_buf.clear()
                        pres_buf.clear()
                        continue

                # 识别模式下正常检测
                if mode == "recognize":
                    # 防抖确认词
                    confirmed: str | None = None
                    if label >= 0:
                        if label == stable_label:
                            stable_count += 1
                        else:
                            stable_label = label
                            stable_count = 1
                        if (stable_count == STABLE_N
                                and (label != last_label or now - last_label_time > COOLDOWN_S)):
                            confirmed = VOCAB[label]
                            last_label = label
                            last_label_time = now
                    else:
                        stable_label = -1
                        stable_count = 0

                    if confirmed:
                        sentence_words.append(confirmed)
                        full = "".join(sentence_words)
                        print(f"[{time.strftime('%H:%M:%S')}] + {confirmed}  ->  {full}  ({conf:.2f})")

                    # 断句
                    sentence_ended = False
                    if sentence_words and time.time() - last_hand_time > SENTENCE_TIMEOUT:
                        cleaned = cleanup_sentence(sentence_words)
                        full_sentence = "".join(cleaned)
                        if cleaned != sentence_words:
                            print(f"[{time.strftime('%H:%M:%S')}] clean: {''.join(sentence_words)} -> {full_sentence}")
                        print(f"[{time.strftime('%H:%M:%S')}] >>> {full_sentence}")
                        if tts:
                            tts.speak(full_sentence)
                        push_to_p4(full_sentence)
                        sentence_words.clear()
                        sentence_ended = True

                # 识别模式: 屏幕显示检测信息; 待机/退出询问: 纯画面
                if mode == "recognize":
                    vis = landmarker.draw(frame, kp, presence)
                    line1 = f"{VOCAB[label]} ({conf:.2f})" if label >= 0 else "(detecting...)"
                    vis = put_text_cn(vis, line1, (10, 10), font_size=32, color_bgr=(0, 255, 255))
                    sentence_text = "".join(sentence_words) if sentence_words else ""
                    vis = put_text_cn(vis, sentence_text, (10, 50), font_size=30, color_bgr=(255, 255, 255))
                    cv2.putText(vis, f"hands: {num_hands}/2  fps: {fps:.1f}",
                                (10, vis.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
                    cv2.imshow("infer", vis)
                else:
                    cv2.imshow("infer", frame)

            # FPS
            fps_n += 1
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_n = 0
                fps_t = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = True
            if key == ord("m"):
                if mode == "idle":
                    mode = "recognize"
                    push_to_p4("进入手语识别")
                    print(f"[{time.strftime('%H:%M:%S')}] >>> 进入识别 (手动)")
                elif mode == "voice":
                    mode = "idle"
                    wake_blocked_until = now + 3.0
                    reduce_camera_quality(30)  # 恢复画质
                    print(f"[{time.strftime('%H:%M:%S')}] <<< 返回主界面")
                else:
                    mode = "idle"
                    sentence_words.clear()
                    kp_buf.clear()
                    pres_buf.clear()
                    push_to_p4("已退出识别")
                    if network_time:
                        cur_dt = network_time + timedelta(seconds=now - last_ntp_fetch)
                    else:
                        cur_dt = datetime.now()
                    push_to_p4_urgent(f"{WEATHER_CITY} {weather_str}\n{cur_dt.strftime('%m/%d %H:%M')}\n比1进入手语识别\n比2进入语音识别")
                    print(f"[{time.strftime('%H:%M:%S')}] <<< 返回待机 (手动)")

    cv2.destroyAllWindows()
    landmarker.close()
    if tts:
        tts.stop()


if __name__ == "__main__":
    main()
