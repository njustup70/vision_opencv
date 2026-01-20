# 二维码通信

## CameraNode 类（camera_node.py）摄像头节点，发布图像
### 发布话题：camera/image_raw
### 参数配置: camera_index：摄像头设备号、brightness、contrast、exposure：图像参数、fps：帧率

## qr_core.py 二维码编码/解码核心类QRCoder，前24位为12个位置的KFS状态，预留后8位未编码
### 编码`encode(states, size_cm, dpi, save_dir)` 
### 输入：12个状态列表（如 ["空","R1","假",...],输出：二维码图片路径、对应的16进制字符串
### 解码`decode(hex_str)` 
### 输入：16进制字符串（8位,输出：12个状态的列表
```python
from qr_core import QRCoder
path, hex_data = QRCoder.encode(states_list, size_cm=15, dpi=220)
states = QRCoder.decode(hex_data)
```

## QRDetectNode 类（qr_detect_node.py）ROS 2 节点，用于二维码检测
### 订阅话题：camera/image_raw，QRDetectNode.show_qr(img, duration_ms)：在便携屏全屏显示二维码
### 检测结果通过 detected_data 获取