import deform_restore as drs
import cv2
import numpy as np
import os

loc = []
img_name = "5.jpg"
img_path = "photos/" + img_name
os.makedirs("restored_images", exist_ok=True)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        loc.append((x, y))
        if len(loc) == 4:
            print("Four points selected, starting restoration...")
            # 顺时针将点排序
            center = np.mean(loc, axis=0)
            sorted_points = sorted(loc, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
            restore_img = drs.ROIRestore(img, np.array(sorted_points, dtype=np.float32), image_shape=[500,500])
            cv2.imshow("Restored Image", restore_img)
            cv2.waitKey(0)
            cv2.destroyWindow("Restored Image")
            loc.clear()
            result_path = "restored_images/restored_" + img_name
            ok = cv2.imwrite(result_path, restore_img)
            if ok:
                print(f"Restored image saved to {result_path}")
            else:
                print("Failed to save restored image.")

img = cv2.imread(img_path)
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Original Image", mouse_callback)
cv2.imshow("Original Image", img)
cv2.waitKey(0)  
cv2.destroyAllWindows()
