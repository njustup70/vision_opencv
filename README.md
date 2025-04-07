### 这里是计算机视觉部分的代码测试库 目前大致有三个方向：
#### 1.基于深度学习(YOLOv11)的目标检测与跟踪
#### 2.基于ArUco码的机器人视觉跟踪定位
#### 3.基于Pnp的相机位姿解算
### - YOLOv11板块时间戳:
#### 2024-11-25(wxm-dev) : 先使用通用目标检测数据集coco模型对模型预训练，后使用自定义的篮球数据集进行训练，权重文件保存在 "YOLOv11/models/best-20241124.pt"，训练的详细参数及可视化见[这里](https://www.comet.com/summerwen-lab/basketball-recognition/8bdde0faff8848929aa2f45d74f56469?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=epoch)

#### 2024-11-21(wxm-dev) ：上传了YOLOv11官方文档中文版Train部分，并附带了一个可以通过Comet_TL监督训练过程的示例训练程序和数据集配置文件("YOLOv11/demo_train.py YOLOv11/demo_train.yaml")
#### 2024-11-19(wxm-dev) : 上传了完整的可以在Docker容器里运行的YOLOv11版本，并附带了一个简单的使用例程代码("YOLOv11/demo_predict.py")
#### 2024-11-18(wxm-dev) : 上传了YOLOv11官方文档中文版Predict部分,在"YOLOv11官方介绍文档中文版"下。

### - ArUco板块时间戳：
#### 2025-03-19(wxm-dev) : 上传了基础的针对图片的ArUco码译码脚本以及一个简单的ArUco码生成脚本（用于快速测试）
#### 2025-03-19(wxm-dev) : 将原始代码封装成了ArUco类，支持图像和视频检测以及快速ArUco码生成。以及提供了一个ArUco类的使用示例脚本。

### - Pnp位姿解算板块时间戳：
#### 2025-03-27(wxm-dev) : 上传了相机内参和畸变系数标定的代码（包含黑白棋盘格图片生成和相机标定计算）
