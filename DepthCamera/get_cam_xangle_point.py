# 相机俯角测量
# 只包含单点的平面法向量测量
import numpy as np
from plan_PC_fit import region_growing_plane, depth_to_point_cloud

def fit_plane_from_depth(depth_img, camera_model, u, v, depth_scale=1.0, points = None):
    """
    从深度图中指定像素(u,v)出发，找到该平面法向量
    """
    # 转点云
    if points is None:
        pcd, valid_mask = depth_to_point_cloud(depth_img, camera_model, depth_scale)
        h, w = depth_img.shape
        if not valid_mask[int(v), int(u)]:
            raise ValueError("该点无效或深度为0")

    #直接新加点,平均2.2s
    target_point = np.array([
        (u - camera_model.cx()) * depth_img[int(v), int(u)] / (camera_model.fx() * depth_scale),
        (v - camera_model.cy()) * depth_img[int(v), int(u)] / (camera_model.fy() * depth_scale),
        depth_img[int(v), int(u)] / depth_scale
    ])
    pcd.points.append(target_point)
    seed_idx = len(pcd.points) - 1

    # 区域向量
    _, plane_model, _, _, _ = region_growing_plane(pcd, seed_idx)
    seed_normal = np.array(plane_model[:3])
    if seed_normal[2] > 0:
        seed_normal = -seed_normal  # 保持法向量朝着相机

    return {
        "seed_normal": seed_normal
    }
