# 小板 UART 通信协议（视觉 -> STM32）

本文档定义视觉系统向 STM32 发送 **小 ChArUco 码盘左右/上下偏移** 的最小串口协议。  
格式尽量与 `serial_dispose` 模板保持一致（固定帧头 + 固定帧 ID + 固定长度 payload）。

## 1) 串口参数
- 波特率：115200
- 数据位：8
- 校验位：无
- 停止位：1
- 字节序：小端（Little-endian）
- 端口类型：USB-TTL （CH340）

## 2) 帧结构（PC -> STM32）

```
[SOF][ID][PAYLOAD...]
SOF:  1 字节，固定 0xFA
ID:   1 字节，帧类型
PAYLOAD: 固定长度，长度由 ID 决定
```

该最小版本 **不包含长度字段与 CRC 校验**。

## 3) 帧定义：小板偏移（核心帧）

**用途：** STM32 根据 left/up 偏移控制机械臂运动。

- SOF：`0xFA`
- ID：`0xB1`（如与现有 ID 冲突，请改为未使用的值）
- PAYLOAD：`<ff`（2 个 float32，小端）
  - `left_mm`（float32）
  - `up_mm`（float32）

**总长度：** 1 + 1 + 8 = 10 字节

## 4) 坐标与符号约定

视觉节点使用相机光学坐标系（OpenCV 约定）：
- x 向右，y 向下，z 向前

偏移定义如下：
- `left_mm = -x * 1000`
- `up_mm   = -y * 1000`

如果电控方向相反，可在桥接节点参数中翻转符号：
- `invert_left`
- `invert_up`

## 5) 质量门控（PC 侧）

桥接节点可按质量阈值决定是否发送：
- `min_confidence`
- `max_mean_reproj_px`

若门控失败，则不发送该帧；STM32 侧应设置**超时保护**（例如 200ms 未收到就停机/保持）。

## 6) 示例帧

```
FA B1 | <ff payload>
```

举例：
- `left_mm = +12.500`
- `up_mm = -3.250`

payload 对应 `struct.pack("<ff", 12.5, -3.25)`。

## 7) ROS 节点与启动方式

- 节点：`small_board_serial_bridge`
- 启动文件：`small_board_serial_bridge.launch.py`

示例：
```
ros2 launch spear_vision small_board_serial_bridge.launch.py \
  port:=/dev/serial_ch340 baudrate:=115200 out_frame_id:=177
```

