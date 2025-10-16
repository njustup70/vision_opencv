from ultralytics import YOLO
# 记得切换对应路径

model = YOLO("YOLOv11/models(TrainedByMyself)/Garbage.pt")

model.predict(
    source = "YOLOv11/ultralytics/assets/garbage",
    show = True,
    save = True
)