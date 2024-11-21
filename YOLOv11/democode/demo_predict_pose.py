from ultralytics import YOLO

model = YOLO("YOLOv11/models(TrainedByMyself)/demo_predict_pose.pt")

video_source = "YOLOv11/ultralytics/assets/test_vedio/test2.mp4"

results = model.predict(
    source = video_source,
    imgsz = 480,
    device = 0,
    # stream = True,  
    save = True,  
    show = True 
)

