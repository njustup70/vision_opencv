# VERSION_OPENCV
## RC2026：这是2026赛季的视觉仓库，主要存放yolo，opencv，pnp解算和arucuo识别等代码。
## 维护者：陈成  &nbsp; QQ:194025781 
## 容器构建
### 当前支持ROS构建和GPU构建镜像两种方式，其中GPU构建主要是在本地训练模型，ROS构建主要是后期部署
- 使用历程
```bash
(gpu)|.devcontainer/gpu$ ./build.sh 
```
```bash
(ros)|.devcontainer/ros$ ./build.bash
```
### 相应的容器名称后缀不同 yolo_container_cpu / yolo_container_gpu
## 模块介绍
|模块 |说明 |
|---|---|
|[CameraCalibration](./CameraCalibration/)|相机标定|
|[cv_lib](./cv_lib/)|opencv,yolo,aruco,pnp库函数存放|
|[yolo](./yolo/)|yolo模型训练|
### 对应三个文件夹下的均设置对应readme，可参照进行使用


## 问题
1.遇到显示界面崩溃问题的  
暂定用如下方式，有点纯（后续优化）——补足部分依赖
```bash
sudo apt-get install nautilus
```
2.遇到无法找到模块解析路径问题(已解决)
```bash
export PYTHONPATH="/home/Elaina/yolo:$PYTHONPATH"
```