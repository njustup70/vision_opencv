# 导入所需的库
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import time
import cv2
import numpy as np
from cv_lib.ros.cv_bridge import ImagePublish_t, ImageReceive_t
from PoseSolver.Aruco import Aruco
from PoseSolver.PoseSolver import PoseSolver
from YOLOv11.yolo_lib import MyYOLO
#from yolo_test import MyYOLO

def main():
    # 相机内参矩阵
    camera_matrix = np.array([
        [606.634521484375, 0, 433.2264404296875],
        [0, 606.5910034179688, 247.10369873046875],
        [0.000000, 0.000000, 1.000000]
    ], dtype=np.float32)
    
    # 畸变系数
    dist_coeffs = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
    
    # 创建处理管道
    pipe = []
    pipe.append(ImageReceive_t(print_latency=True))
    pipe.append(MyYOLO("/home/Elaina/yolo/best.pt", show=True))
    pipe.append(ImagePublish_t("yolo"))
    
    # 添加位姿解算器，设置print_result为True以打印位姿信息
    pose_solver = PoseSolver(camera_matrix, dist_coeffs, marker_length=0.1, print_result=True)  # 假设标记长度为0.1米
    
    content = {}
    print_time = True
    
    try:
        while True:
            # 初始化图像（实际应用中应从相机获取）
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 处理管道
            for p in pipe:
                try:
                    if print_time:
                        start_time = time.time()
                    
                    p.update(image, content)
                    
                    if print_time:
                        end_time = time.time()
                        print(f"{type(p).__name__}: {(end_time - start_time)*1000:.2f} ms")
                
                except Exception as e:
                    print(f"处理模块 {type(p).__name__} 出错: {str(e)}")
                    continue
            
            # 如果有检测到目标，进行位姿解算
            if "corners" in content and len(content["corners"]) > 0:
                print("\n" + "="*50)
                print(f"检测到 {len(content['corners'])} 个目标，开始位姿解算...")
                
                # 使用PoseSolver的update方法处理所有目标
                pose_solver.update(image, content)
                
                # 打印位姿信息
                if "pnp" in content and content["pnp"]:
                    # 注意：根据当前PoseSolver实现，content["pnp"]是单个字典，不是列表
                    pose_info = content["pnp"]
                    
                    for i, corner_data in enumerate(content["corners"]):
                        conf = corner_data["confidence"]
                        
                        # 只处理第一个目标（因为当前PoseSolver只处理第一个）
                        if i == 0:
                            print(f"\n目标 {i+1} (置信度: {conf:.2f}):")
                            print(f"  Yaw: {pose_info['yaw']:.1f}°, Pitch: {pose_info['pitch']:.1f}°, Roll: {pose_info['roll']:.1f}°")
                            print(f"  位置: [{pose_info['tvec'][0][0]:.3f}, {pose_info['tvec'][1][0]:.3f}, {pose_info['tvec'][2][0]:.3f}]m")
                            print(f"  距离: {pose_info['distance']:.3f}m")
                        else:
                            print(f"\n目标 {i+1} (置信度: {conf:.2f}):")
                            print("  位姿未计算（当前只处理第一个目标）")
                
                print("="*50 + "\n")
            
            # 显示结果
            cv2.imshow("Detection Result", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        cv2.destroyAllWindows()
        print("程序结束")

if __name__ == "__main__":
    main()    