from ultralytics import YOLO
import cv2

model = YOLO("1.20_openvino_model/")

cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame)
    
    annotated_frame = results[0].plot()
    
    cv2.imshow("OpenVINO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()