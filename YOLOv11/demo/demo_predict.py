from ultralytics import YOLO

model = YOLO("YOLOv11/models(TrainedByMyself)/Garbage.pt")

model.predict(
    source = "YOLOv11/ultralytics/assets/garbage",
    show = True,
    save = True
)