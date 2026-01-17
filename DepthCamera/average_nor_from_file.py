# 若数据点有异常需改阈值，包括delta_angle、删点比例和最少点下限等，可用此脚本重新计算平均法线
import numpy as np

rotation = [0.9999970197677612, -0.00011402424570405856, -0.0024283782113343477,
           0.0001164410641649738, 0.9999995231628418, 0.0009951215470209718,
           0.0024282634258270264, -0.0009954014094546437, 0.999996542930603]

normal_list = []
file_path = "DepthCamera/xangle.txt"
flag = 0
with open(file_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("2026-01-13 08:04:08"):
            flag = 1
            continue
        if flag == 1 and line.startswith("Average normal vector"):
            flag = 0
            break
        if flag == 1 and not line.strip():
            flag = 0
            break
        if flag == 1:
            parts = line.strip().split(",")
            normal = [float(part) for part in parts]
            normal_list.append(normal)

all_points = len(normal_list)
normal_array = np.array(normal_list)
normal_array_original = normal_array.copy()
average = np.mean(normal_array, axis=0)
average /= np.linalg.norm(average)  # 单位化

delta_angle = 0.5
flag = 1
while flag:
    normal_array = normal_array_original.copy()
    average_good = average.copy()
    cos_thresh = np.cos(np.deg2rad(delta_angle))
    angle_error = 0
    times = [0, 0]
    while average_good @ average < cos_thresh or times[0] == 0:
        average = average_good.copy()
        dots = normal_array @ average
        k = int(0.7 * len(dots))               # 取最近 70%
        idx = np.argsort(dots)[-k:]
        normal_array_good = normal_array[idx]   #先不删
        average_good = np.mean(normal_array_good, axis=0)
        average_good /= np.linalg.norm(average_good)
        times[0] += 1
        if times[0] > 100:   # 防止死循环
            angle_error = 1
            delta_angle += 0.1
            break
    if angle_error:
        continue

    average = average_good.copy()
    # 先删一次
    dots = normal_array @ average
    k = int(0.7 * len(dots))
    idx = np.argsort(dots)[-k:]
    normal_array = normal_array[idx]
    while average_good @ average < cos_thresh or times[1] == 0:
        average = average_good.copy()
        dots = normal_array @ average
        k = int(0.7 * len(dots))
        idx = np.argsort(dots)[-k:]
        normal_array = normal_array[idx] # 删
        if len(normal_array) < 10:
            break
        average_good = np.mean(normal_array, axis=0)
        average_good /= np.linalg.norm(average_good)
        times[1] += 1
        if times[1] > 100:  # 防止死循环
            break
    else:
        flag = 0
        break

    delta_angle += 0.1 # 放宽条件，继续迭代,以防过小时无结果
    if delta_angle > 5:
        print("Failed to converge average normal vector within reasonable angle threshold.")
        average = None
        average_good = None
        break

average_good = np.array(rotation).reshape(3,3) @ average_good.reshape(3,1)
average_good = average_good.reshape(3,)
print(f"Average normal vector: {average_good[0]:.6f},{average_good[1]:.6f},{average_good[2]:.6f}\nPoints used: {len(normal_array)} in all points {all_points}\n")
print(f"Iterations without deletion: {times[0]}, with deletion: {times[1]}, delta_angle: {delta_angle}")