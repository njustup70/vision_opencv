from ultralytics import YOLO
import cv2
import os

model = YOLO("YOLOv11/models/best_20241124.pt")

target_class = 0  # basketball

results = model.predict(
    source="YOLOv11/ultralytics/assets/basketball/test_badlight",  
    imgsz=640,      
    device=0,       
    save=False,    
    show=False,   
)


output_path = "YOLOv11/runs/detect/predict/basketball_badlight"
os.makedirs(output_path, exist_ok=True)  

for result in results:
    img = result.orig_img 
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls)  
            if cls == target_class:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf  

                label = f"Basketball {conf.item():.2f}"

                rectangle_color = (0, 0, 256)  # (B, G, R)
                line_thickness = 3  # 设置线条粗细

                cv2.rectangle(img, (x1, y1), (x2, y2), rectangle_color, thickness=line_thickness)

                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)

    filename = os.path.basename(result.path) 

    cv2.imwrite(os.path.join(output_path, filename), img)

    # cv2.imshow("Sports Ball Detection", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
