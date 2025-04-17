## 与 ROS 2 通信的两种方式

### 1. 共享内存 + ZMQ 通信方式

- **优点**：传输速度快  
- **缺点**：扩展性较差，接入复杂  
- **适用场景**：图像流传输等大数据量通信  
- **使用说明**：需要在ros2 driver容器启动 `image_bridge.py` 脚本

### 2. 使用 roslibpy 通信方式

- **优点**：API 类似于 ROS 1，易于上手  
- **缺点**：通信速度较慢  
- **适用场景**：小数据传输，如点、标量数据等  
- **使用说明**：需要在ros2容器启动 `rosbridge_server`  
```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```
### 3.测试方法
打开ros/cv_bridge.py

如果成功收到图片会打开gui显示

## 相关链接:
- [roslibpy文档](https://roslibpy.readthedocs.io/en/latest/)