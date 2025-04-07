import cv2
import numpy as np
import glob
import yaml

# ================== 参数配置 ==================
CHECKERBOARD = (9, 6)        # 棋盘格内角点数量 (cols, rows)
SQUARE_SIZE = 0.02348       # 棋盘格实际边长（米）

CALIB_IMG_PATH = "./calibration2/*.jpg"  # 标定图片路径
OUTPUT_FILE = "./yamls/camera_calibration.yaml"  # 标定结果保存路径


# ================== 标定流程 ==================
def calibrate_camera():
    # 【1】准备3D世界坐标点 (z=0)
    obj_points = []
    for i in range(CHECKERBOARD[1]):
        for j in range(CHECKERBOARD[0]):
            obj_points.append([j*SQUARE_SIZE, i*SQUARE_SIZE, 0])
    objp = np.array(obj_points, dtype=np.float32)

    # 【2】存储所有检测到的角点
    img_points = []  # 2D图像点
    obj_points_list = []  # 3D世界点
    images = glob.glob(CALIB_IMG_PATH)
    image_size = None

    # 【3】遍历所有标定图片
    for fname in images:
        img = cv2.imread(fname)
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])  # (width, height)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 【4】查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            # 【5】亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            obj_points_list.append(objp)
            img_points.append(corners_refined)
            
            # 【6】可视化角点
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # 【7】执行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, img_points, image_size, None, None)

    # 【8】计算重投影误差
    mean_error = 0
    for i in range(len(obj_points_list)):
        img_points_proj, _ = cv2.projectPoints(
            obj_points_list[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2)/len(img_points_proj)
        mean_error += error
    print(f"平均重投影误差: {mean_error/len(obj_points_list):.2f} 像素")

    # 【9】保存标定结果 
    """
        camera_matrix:
        - [fx, 0, cx]
        - [0, fy, cy]
        - [0, 0, 1]

        dist_coeffs:
        - [k1, k2, p1, p2, k3]
    """
    data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "reprojection_error": float(mean_error/len(obj_points_list))
    }
    with open(OUTPUT_FILE, 'w') as f:
        yaml.dump(data, f)
    
    print("标定完成！结果已保存至", OUTPUT_FILE)
    return mtx, dist

if __name__ == "__main__":
    calibrate_camera()