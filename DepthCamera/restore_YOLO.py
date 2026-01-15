# YOLO目标检测模块
# 基于变形恢复图像预处理, 将恢复后图像输入YOLO进行目标检测
from ultralytics import YOLO
import deform_restore as rs
import numpy as np

target_loc=(0,100,500)
target_size=(500,500) # ROI大小,调大一点，不知道为什么YOLO只能识别占比小一点的

def img_preprocess(img, depression_angle, target_loc, target_direct = 0, target_size = (500,500)): # 默认看正面
    """
    图像进入YOLO的预处理
    
    :param img: 原图像 :type:`np.ndarray`
    :param depression_angle: 相机俯仰角 (nx, ny, nz) :type:`list`
    :param target_loc: 目标3D位置 (x, y, z)
    :param target_direct: 目标方向，默认看正面，看上面传1 :type:`int`
    :param target_size: 目标大小，默认(500,500) :type:`tuple`
    """
    
    up_normal = np.array(depression_angle).reshape(3,) # 水平面(上面)法向量
    forward_normal = np.cross(up_normal, np.array([1,0,0])) # 前方向法向量
    if target_direct == 0: # 看正面
        plane_normal = forward_normal
        up_direct = up_normal
    else: # 看上面
        plane_normal = -up_normal
        up_direct = forward_normal

    # 计算平面旋转rot矩阵
    
    roi_img, roi_2d = rs.deformRestore(img, target_loc,(target_size[0], target_size[1], plane_normal.reshape(3,)), up_direct=up_direct, image_shape=target_size)
    return roi_img, roi_2d

def get_yolo_result(model, img):
    result = model.predict(source=img, save=False, save_txt=False, conf=0.005, iou=0.45)
    return result
