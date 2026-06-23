# 手语识别与语音交互系统

## 目录结构

```
model_and_run/
├── ml/                    # Python 端：模型训练/推理 + OLED 驱动
│   ├── infer.py           # 主程序入口
│   ├── config.py          # 全局配置
│   ├── model.py           # 神经网络结构
│   ├── train.py           # 训练脚本
│   ├── dataset.py         # 数据加载
│   ├── preprocess.py      # 关键点预处理
│   ├── record.py          # 数据采集
│   ├── hands.py           # MediaPipe 手部检测
│   ├── stream_reader.py   # 摄像头流读取
│   ├── draw_cn.py         # 中文渲染到画面
│   ├── tts.py             # 语音播报
│   ├── best.pt            # 训练好的模型权重
│   ├── oled.c / .h        # P4 OLED 驱动 (二分查找+异步队列)
│   ├── font.h             # 中文字
└── main/                  # ESP32-P4 固件
    ├── simple_video_server_example.c  # HTTP 服务 + /api/control
    └── CMakeLists.txt     # 编译配置
```

---

## ml/ 模块说明

### 主程序

**infer.py** — 程序入口，运行 `python infer.py --source [摄像头URL或0]`

三个工作模式：
- 待机：OLED 显示时间温度，摄像头纯画面，手指计数
- 手语识别（比1进入）：MediaPipe→CNN→防抖确认→句子累积→TTS播报→OLED显示
- 语音识别（比2进入）：发送"985"指令给 P4 启动麦克风，自动降画质

核心参数：`CONF_THRESH=0.85`（置信度阈值）、`STABLE_N=8`（连续确认帧数）、`SENTENCE_TIMEOUT=3.5s`（断句超时）

### 配置

**config.py** — 全局常量，改一处全局生效

| 项 | 说明 |
|---|---|
| VOCAB | 17 词词汇表，索引即类别 ID |
| SEQ_LEN/BUF_LEN | 模型输入 30 帧，缓冲区 60 帧 |
| FEATURE_DIM | 128 维特征 (2手×21点×3坐标+2标志) |
| BATCH_SIZE/LR/EPOCHS | 训练超参 |
| WAKE_WORD | 唤醒词 "请" |
| IDLE_TIMEOUT | 待机超时 10 秒 |

### 模型

**model.py** — SignNet：轻量 1D-CNN

```
输入 (B, 128, 30)
  → Conv1d → BN → ReLU → Conv1d → BN → ReLU → MaxPool (30→15)
  → Conv1d → BN → ReLU → Conv1d → BN → ReLU → AdaptiveAvgPool
  → Flatten → Dropout(0.3) → Linear(128→17)
```

参数量 ~114K，AdaptiveAvgPool 使模型能适应不同序列长度。

**best.pt** — 训练好的权重，验证准确率 99.2%。

### 训练

**train.py** — 从头训练，运行 `python train.py`

- 加权交叉熵损失（样本数少的类别权重大）
- CosineAnnealingLR 学习率衰减
- 自动保存 best.pt（验证最优）和 last.pt（最后一轮）
- 训练日志输出到 checkpoints/train_log.csv

**dataset.py** — PyTorch Dataset，从 `data/` 目录加载 `.npz` 文件。支持数据增强（随机抖动、缩放）。`split_by_label()` 按标签分层划分训练/验证集。

**preprocess.py** — 关键点预处理：
1. 基于锚手（出现帧数多的那只手）的手腕平移对齐
2. 缩放到单位标准差归一化
3. 双手 21×3 坐标拉成 126 维 + 2 维 presence = 128 维

### 数据采集

**record.py** — 录制手势数据，运行 `python record.py --source 0`

- 数字键 0-9 + `n`/`p`/`[`/`]` 切换词
- 空格开始/停止录制，录满 30 帧自动保存
- 输出 `.npz` 文件到 `data/` 目录

### 辅助模块

**hands.py** — 封装 MediaPipe Hands。`detect(frame)` 返回双手 (2, 21, 3) 关键点坐标和 (2,) presence 标志。`draw()` 绘制骨架。

**stream_reader.py** — 统一适配本地摄像头和 HTTP MJPEG 流。后台线程持续抓帧，主线程取最新帧（丢旧帧保证低延迟）。支持 `cv2.VideoCapture`（本地）和 `requests` 拉流（远程）。

**draw_cn.py** — 用 PIL 渲染中文字符到 OpenCV BGR 图像上，解决 `cv2.putText` 无法显示中文的问题。

**tts.py** — Windows SAPI 中文语音播报，`pyttsx3` 实现，独立线程避免阻塞主循环。

---

## main/ P4 固件

**simple_video_server_example.c** — ESP32-P4 主固件

- HTTP 服务器：静态文件前端 / MJPEG 推流 / REST API
- `/api/control` 端点：接收 `{"hex":"GBK编码"}` 显示到 OLED
- `/api/set_camera_config`：调整 JPEG 质量

**CMakeLists.txt** — ESP-IDF 编译配置，依赖 `oled.c`、`font.h` 和前端 gzip 资源。

---

## OLED 驱动

**oled.c** — SSD1306 I2C 驱动，关键优化：

- **二分查找汉字**：O(log n) 替代原 O(n) 线性搜索
- **FreeRTOS 队列异步刷新**：`ssd1306_show_text()` 消息入队即返回，独立 Task 消费，不阻塞 HTTP 线程
- **支持 `\n` 换行、`\r` 回车**

**oled.h** — 接口声明：`ssd1306_init()`、`ssd1306_show_text()`、`oled_start_task()`

**font.h** — 中文字库 (GBK, 16×16 点阵)

---

## 运行方式

```bash
# 安装依赖
pip install opencv-python mediapipe torch numpy requests pyttsx3

# 本地摄像头
cd ml && python infer.py --source 0

# 远程 P4 摄像头
cd ml && python infer.py --source http://10.244.3.204:81/stream

# 录制数据
python record.py --source 0

# 训练
python train.py
```
