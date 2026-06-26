import cv2

# 全局变量存储点击的坐标
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"记录点: [{x}, {y}]")
        # 在图上画个圆圈标记
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)

# 假设你读取了一张当前固定机位的图片
img = cv2.imread('saves/detect_0000.jpg') 
cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()