# 深度相机
## 包含了深度相机相关的功能实现及测试代码
### PlaneDepToSpace.py
 - 功能：识别框内主体深度并对主体深度块计算重心及重心对应的空间相对坐标
 - 使用方式：
```
self.depth_camera = DepthCamera()
self.depth_camera.loadCameraInfo(info_d = None, info_c = None, info_d2c = None)
center = self.depth_camera.depthImageFindCenter(self.range, self.img)
```
 - loadCameraInfo可以传入从/camera/(depth & color)/info读入的内参和/camera/depth_to_color话题的外参，不传将从文件读取（其实没区别）
 - range[0],range[1]为左上角像素坐标，range[2],range[3]为右下角
 - 采用深度图与彩图直接叠加方案，所以需要提前将color_raw resize为depth_raw或者把框的range分别除以1.5 （1280 * 720 -> 848 * 480）
 - 传入的img为话题/camera/depth/image_raw数据
 - 返回值为重心的像素坐标uv和深度、主体深度块占比、重心相对坐标[x,y,z]
 - 相对坐标为相机坐标系（有关orbbec相机坐标系详见[4.4.坐标系和TF变换 — OrbbecSDK V2 ROS2封装文档](https://orbbec.github.io/OrbbecSDK_ROS2/zh/source/4_application_guide/coordinate_and_tf.html)）
 - 后半部分为测试
 - 测试时可分别点击画面所需部分对角进行手动框选

### SpaceToPlaneDep.py
 - 仅包含测试代码
 - 用于将/camera/depth/points点云话题映射到深度图并进行误差的测试
 - 根据测试该点云就是深度图的映射
 - 目前不进行点云数据的处理