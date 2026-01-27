import cv2
import yaml
import subprocess

file_name = "USB_capture.yaml"
file_path = "cv_lib/" + file_name

with open(file_path, "r") as f:
    cfg = yaml.safe_load(f)

cam = cfg["camera"]
ctrls = cam["controls"]

dev = cam["device"]

def set_ctrl(name, value):
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", dev, "-c", f"{name}={value}"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to set control {name} to {value}: {e}")
        raise

# 下发所有控制参数
for k, v in ctrls.items():
    set_ctrl(k, v)


#视频（摄像头）
cv2.namedWindow("result",cv2.WINDOW_FREERATIO)
cv2.resizeWindow("result",640,480)
vc = cv2.VideoCapture(dev, cv2.CAP_V4L2)
vc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#vw= cv2.VideoWriter("output.mp4",fourcc,30,(640,480))

while vc.isOpened():
    ret,frame = vc.read()
    if not ret:
        break
    # frame = cv2.flip(frame,1)       #左右镜像
    #vw.write(frame)
    cv2.imshow('result',frame)
    if cv2.waitKey(1) == 27:
        break
    
vc.release()
#vw.release()
cv2.destroyAllWindows()