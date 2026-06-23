| 支持的芯片 | ESP32-P4 | ESP32-S3 | ESP32-C3 | ESP32-C6 | ESP32-C5 |
|-----------|----------|----------|----------|----------|----------|

# 简易视频服务器示例

## 概述

本例程演示了如何在局域网中使用多个端口创建 HTTP 服务器。通过网页浏览器即可访问视频流和图像采集功能。

## API 端点

| 端口 | 端点 | 方法 | 说明 |
|:----:|:---------|:------:|:------------|
| 80 | `/` | GET | 提供浏览器端视频展示的主 HTML 页面 |
| 80 | `/api/capture_image?source={n}` | GET | 从指定摄像头获取 JPEG 图像。<br/>**参数**: `n` — 摄像头编号（0=第一个，1=第二个） |
| 80 | `/api/capture_binary?source={n}` | GET | 从指定摄像头获取原始二进制图像数据。<br/>**参数**: `n` — 摄像头编号 |
| 80 | `/api/get_camera_info` | GET | 获取所有摄像头信息，包括分辨率和 JPEG 压缩设置 |
| 80 | `/api/set_camera_config` | POST | 配置摄像头参数，包括分辨率和 JPEG 压缩质量 |
| 81 | `/stream` | GET | **第一个**摄像头的连续 MJPEG 视频流 |
| 82 | `/stream` | GET | **第二个**摄像头的连续 MJPEG 视频流 |

> **注意**: 服务器在后台持续向客户端推送 JPEG 图像流。从网页端保存图像时，保存的图片可能不是实时数据。

### 域名访问

默认情况下开启了 mDNS（组播 DNS），可用域名代替 IP 地址访问：
- 图像采集：`http://esp-web.local/api/capture_image?source=0`
- 主界面：`http://esp-web.local`

也可以直接使用设备 IP 地址访问。

## 快速上手

### 硬件配置

使用前请参考[视频初始化配置指南](../common_components/example_video_common/README.md)了解：

- 开发板级配置
- 摄像头接口设置
- GPIO 引脚分配
- 时钟频率设置

### 项目配置

打开配置菜单：

```bash
idf.py menuconfig
```

#### 网络连接设置

**WiFi 接口配置:**
- **WiFi SSID 和密码**: ESP32 连接网络必需
- **SoftAP 设置**: 如需要 ESP32 作为热点使用

**以太网接口配置:**
- **PHY 型号**: 在 `Ethernet PHY` 选项中选择（如 IP101）
- **PHY 地址**: 根据原理图设置
- **时钟配置**: 配置 EMAC 时钟模式和 SMI GPIO 引脚

#### 摄像头传感器配置

在 **Espressif Camera Sensors Configurations** 菜单中：
- 选择要使用的摄像头传感器
- 选择传感器输出格式

#### 示例专属配置

1. **设置目标平台：**
   ```bash
   idf.py set-target esp32p4
   idf.py menuconfig
   ```

2. **配置视频缓冲区：**
   ```
   Example Configuration → Camera video buffer number (默认 2)
   ```
   更多缓冲区提升性能、减少丢帧，但消耗更多内存。高分辨率传感器（如 1080P）建议用 2 个。

3. **设置 JPEG 压缩质量：**
   ```
   Example Configuration → JPEG compression quality (默认 80%)
   ```
   并非所有摄像头支持此设置，不支持时自动使用最近的合法值。

4. **HTTP 和 mDNS 配置：**
   ```
   Example Configuration →
       HTTP part boundary (默认值)
       mDNS instance (默认 web-cam)
       mDNS host name (默认 esp-web)
   ```
   无特殊需求保持默认即可。

5. **摄像头接口选择：**
   本例程会初始化所有启用的摄像头：
   ```
   Example Video Initialization Configuration →
       Select and Set Camera Sensor Interface →
           [*] MIPI-CSI
           [*] DVP
   ```

6. **共享 I2C 总线配置：**
   如果多个摄像头共用 I2C GPIO（如 ESP32-P4-Function-EV-Board V1.5 上 MIPI-CSI 和 DVP 共用）：
   ```
   Example Video Initialization Configuration →
       [*] Use Pre-initialized SCCB(I2C) Bus for All Camera Sensors
           (0) SCCB(I2C) Port Number
           (8) SCCB(I2C) SCL Pin
           (7) SCCB(I2C) SDA Pin
   ```

7. **选择目标摄像头传感器：**
   ```
   Component config → Espressif Camera Sensors Configurations →
       Camera Sensor Configuration →
           Select and Set Camera Sensor →
               选择你的传感器型号
   ```

8. **优化 DVP 接口性能：**
   如需更好的 DVP 帧率：
   ```
   选择传感器后 → Select default output format for DVP interface
       选 JPEG 格式可获得更高帧率
   ```

## 编译和运行

1. **编译烧录：**
   ```bash
   idf.py -p PORT flash monitor
   ```
   （按 `Ctrl-]` 退出串口监视器）

2. 完整说明见 [ESP-IDF 入门指南](https://docs.espressif.com/projects/esp-idf/en/latest/esp32p4/get-started/index.html)。

## 预期输出

正常运行时串口会输出类似以下内容：

```
I (1628) main_task: Started on CPU0
I (5308) example_common: Connected to example_netif_eth
I (5308) example_common: - IPv4 address: 172.168.30.45
I (5318) example_init_video: MIPI-CSI camera sensor I2C port=0, scl_pin=8, sda_pin=7
I (5378) ov2640: Detected Camera sensor PID=0x26
I (5808) example: video0: width=640 height=480 format=RGBP
I (5908) example: Starting stream server on port: '80'
I (5918) example: Camera web server starts
```

## 访问 Web 界面

1. 浏览器打开：
   - `http://esp-web.local`（通过 mDNS）
   - `http://172.168.30.45`（替换为你的设备 IP）

2. **界面功能：**
   - 查看摄像头实时视频流
   - **相机图标**: 下载 JPEG 截图
   - **原始图标**: 下载原始图像数据
   - **齿轮图标**: 配置图像参数

## 常见问题

**I2C 通信错误**

```
E (1595) i2c.master: I2C transaction unexpected nack detected
```

**解决方法:**
- 检查摄像头是否正确连接到开发板
- 检查 menuconfig 中 I2C 引脚 (SCL/SDA) 配置
- 确保板上有 I2C 上拉电阻
- 确认摄像头供电稳定
