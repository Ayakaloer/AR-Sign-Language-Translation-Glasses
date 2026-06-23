"""
全局配置：词汇表、超参、路径。
所有脚本都从这里读，改一个地方就够。
"""
from pathlib import Path

# ---------- 词汇表 ----------
# 索引就是类别 ID。改这个会同步影响数据采集和训练。
VOCAB = [
    "你好",
    "谢谢",
    "再见",
    "对不起",
    "请",
    "我",
    "你",
    "面",      # 原"吃饭", 复用 97 条数据
    "喝水",
    "帮助",
    "饿",
    "来",
    "二",
    "两",
    "今天",
    "天气",
    "好",
]
NUM_CLASSES = len(VOCAB)

# ---------- 视频/关键点参数 ----------
SEQ_LEN = 30          # 模型输入帧数
BUF_LEN = 60          # 缓冲区帧数 (~2秒)，推理时均匀采样 SEQ_LEN 帧喂给模型
NUM_HANDS = 2         # 双手识别
NUM_KEYPOINTS = 21    # MediaPipe Hands 每只手 21 个点
NUM_COORDS = 3        # x, y, z
# 模型输入特征维度: 双手坐标 + 双手 presence 标志
# = 2 * 21 * 3 + 2 = 128
FEATURE_DIM = NUM_HANDS * NUM_KEYPOINTS * NUM_COORDS + NUM_HANDS

# ---------- 训练超参 ----------
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 60
VAL_RATIO = 0.15

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # 录制好的 npz 样本
CKPT_DIR = ROOT / "checkpoints"   # 训练产物
EXPORT_DIR = ROOT / "export"      # ONNX 等导出
for _d in (DATA_DIR, CKPT_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 唤醒与模式 ----------
WAKE_WORD = "请"        # 比划这个词进入手语识别模式, 发送中文到 OLED
IDLE_TIMEOUT = 10.0     # 识别模式下 N 秒没检测到手 → 回到待机界面
WEATHER_CITY = "成都"   # 待机界面天气城市 (wttr.in)

# ---------- ESP32-P4 ----------
P4_HOST = "192.168.1.100"
P4_STREAM_URL = f"http://{P4_HOST}:81/stream"     # camera 0 的 MJPEG 流
P4_TEXT_URL = f"http://{P4_HOST}/api/show_text"   # 推理结果回传 (P4 端待加)
