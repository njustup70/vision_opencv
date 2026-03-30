from ultralytics import YOLO
import cv2

# 加载并检测
model = YOLO("best.pt")
results = model("test.jpg", conf=0.9)

# 处理结果
img = cv2.imread("test.jpg")
shelf = None
kfs = []

for box in results[0].boxes:
    x1,y1,x2,y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    
    if cls == 15:
        shelf = (x1,y1,x2,y2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 3)
    elif cls == 16:
        kfs.append(('红', (x1,y1,x2,y2)))
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    elif cls == 17:
        kfs.append(('蓝', (x1,y1,x2,y2)))
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

# 九宫格分析
if shelf:
    x1,y1,x2,y2 = shelf
    grid = [[[] for _ in range(3)] for _ in range(3)]
    
    for color, (kx1,ky1,kx2,ky2) in kfs:
        cx = (kx1 + kx2) // 2
        cy = (ky1 + ky2) // 2
        i = min(2, (cy - y1) * 3 // (y2 - y1))
        j = min(2, (cx - x1) * 3 // (x2 - x1))
        grid[i][j].append(color)
    
    # 画九宫格
    for i in range(1,3):
        y = y1 + i * (y2-y1)//3
        x = x1 + i * (x2-x1)//3
        cv2.line(img, (x1,y), (x2,y), (0,200,200), 2)
        cv2.line(img, (x,y1), (x,y2), (0,200,200), 2)
    
    # 打印结果
    print(f"检测到: {sum(len(cell) for row in grid for cell in row)}个KFS")
    for i in range(3):
        for j in range(3):
            if grid[i][j]:
                red = grid[i][j].count('红')
                blue = grid[i][j].count('蓝')
                print(f"({i},{j}): {red}红{blue}蓝")

# 保存显示
cv2.imwrite("result.jpg", img)
cv2.imshow("结果", img)
cv2.waitKey(0)
cv2.destroyAllWindows()