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
