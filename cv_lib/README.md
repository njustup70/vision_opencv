# cv_lib
## [aruco_img](./aruco_image/) 
### 存放aruco图片文件夹
## demo
### - [aruco_demo.py](./aruco_demo.py)
### - aruco码识别的简单例子
### - [aruco_img.py](./aruco_image.py)
### - aruco码进行图片识别
## cv_lib
### - [aruco_lib.py](./aruco_lib.py)
```bash
提供类 Aruco 函数，主要负责生成，检测，更新aruco码的内容
```
### - [cv_brigde.py](./cv_bridge.py)
```bash
提供类 ImagePublish_t,ImageSubscribe_t,CompressedImagePublishe_t,CompressedImageSubscribe_t 函数，主要负责接受/发送压缩/未压缩的图像数据
```
### - [cv_mask.py](./cv_mask.py)
```bash
提供类 MaskProcessor 函数，主要负责将掩膜进行优化处理，并提供可视化
```
### - [yolo_lib.py](./yolo_lib.py)
```bash
提供类 MyYOLO 函数，主要执行yolo推理过程，并输出结果
```
### - [PoseSolver.py](./PoseSolver.py)
```bash
提供类 *PoseSolver* 
```


### 关于usb相机udev规则
同一型号多台，VID/PID 一样，甚至 serial 有时为空或重复。
这时用 物理 USB 端口路径 锁死：

先看这台相机插在哪个端口路径：
```bash
udevadm info -a -n /dev/video4 | grep -m1 -E 'KERNELS|DEVPATH'
```

或者更直观：
```bash
readlink -f /sys/class/video4linux/video4/device
```

你会看到类似 .../usb3/3-3.4/... 这种 3-3.4 就是物理路径。

然后规则里加一条（示例，把 3-3.4 换成你实际的）：
```bash
SUBSYSTEM=="video4linux", KERNEL=="video*", \
  ATTRS{idVendor}=="1d6c", ATTRS{idProduct}=="0103", \
  KERNELS=="3-3.4", \
  ENV{ID_V4L_CAPABILITIES}=="*:capture:*", \
  SYMLINK+="usb_camera_1"
```

让规则生效
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=video4linux
```