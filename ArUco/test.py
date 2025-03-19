# This is a example script for "Aruco.py"

from Aruco import Aruco
import cv2

detector = Aruco()

# 返回标注后的图像
annotated_image = detector.detect_image(
    path="test_5x5_100.png",
    aruco_type="DICT_5X5_100",
    if_draw=True
)
cv2.imshow("Result", annotated_image)
cv2.waitKey(0)

# 返回坐标字典
coordinates = detector.detect_image(
    path="test_5x5_100.png",
    aruco_type="DICT_5X5_100",
    if_draw=False
)

for detection in coordinates:
    print(f"ids: {detection['id']}  cx: {detection['cx']}  cy: {detection['cy']}")

detector.detect_video(
    # 这个的终端打印信息被封在了方法的代码里面
    use_camera=True,
    camera_index=1,
    aruco_type="DICT_5X5_100",
    if_draw=False
)