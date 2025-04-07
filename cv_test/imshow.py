import cv2
import numpy as np 
def main():
    # 创建一个全白图像
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255

    cv2.imshow("image", image)  # 显示图像
    cv2.waitKey(0)  # 等待按键事件
    cv2.destroyAllWindows()  # 销毁所有窗口
if __name__ == "__main__":
    main()