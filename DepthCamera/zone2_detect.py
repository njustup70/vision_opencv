import cv2
from ultralytics import YOLO

def main():
    model = YOLO('/home/Elaina/yolo/1.20.pt') 

    camera_index = 4
    cap = cv2.VideoCapture(camera_index)

    while True:
        frame = cap.read()

        results = model(frame, stream=True)

        for result in results:
            annotated_frame = result.plot()

        cv2.imshow("YOLO Real-time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("程序手动终止")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()