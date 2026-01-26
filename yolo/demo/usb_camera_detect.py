from ultralytics import YOLO
import cv2
import subprocess
import os

cmds = [
    ["v4l2-ctl", "-d", "/dev/video11", "--set-ctrl=brightness=0"],
    ["v4l2-ctl", "-d", "/dev/video11", "--set-ctrl=contrast=4"],
    ["v4l2-ctl", "-d", "/dev/video11", "--set-ctrl=saturation=50"],
    ["v4l2-ctl", "-d", "/dev/video11", "--set-ctrl=exposure_time_absolute=300"],
    ["v4l2-ctl", "-d", "/dev/video11", "--set-ctrl=white_balance_automatic=1"],
    ["v4l2-ctl", "-d", "/dev/video11", "--set-ctrl=auto_exposure=3"],
]

for cmd in cmds:
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

os.makedirs("saves", exist_ok=True)

model = YOLO("best.pt")
cap = cv2.VideoCapture(11)

cv2.namedWindow('检测', cv2.WINDOW_NORMAL)
cv2.resizeWindow('检测', 1280, 720)

last_print = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.9, verbose=False)
    
    shelf = None
    kfs = []
    
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            
            if cls == 0:
                shelf = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            elif cls == 1:
                kfs.append(('红', (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif cls == 2:
                kfs.append(('蓝', (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    if shelf:
        x1, y1, x2, y2 = shelf
        grid = [[[] for _ in range(3)] for _ in range(3)]
        
        for color, (kx1, ky1, kx2, ky2) in kfs:
            cx = (kx1 + kx2) // 2
            cy = (ky1 + ky2) // 2
            i = min(2, (cy - y1) * 3 // (y2 - y1))
            j = min(2, (cx - x1) * 3 // (x2 - x1))
            grid[i][j].append(color)
        
        for i in range(1, 3):
            y = y1 + i * (y2 - y1) // 3
            x = x1 + i * (x2 - x1) // 3
            cv2.line(frame, (x1, y), (x2, y), (0, 200, 200), 2)
            cv2.line(frame, (x, y1), (x, y2), (0, 200, 200), 2)
        
        total_kfs = sum(len(cell) for row in grid for cell in row)
        current_print = f"检测到: {total_kfs}个KFS\n"
        
        for i in range(3):
            for j in range(3):
                if grid[i][j]:
                    red = grid[i][j].count('红')
                    blue = grid[i][j].count('蓝')
                    current_print += f"({i},{j}): {red}红{blue}蓝 "
        
        if current_print != last_print:
            print(current_print)
            last_print = current_print
            
            cv2.imwrite(f"saves/detect_{len(os.listdir('saves'))}.jpg", frame)
    
    cv2.imshow('检测', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()