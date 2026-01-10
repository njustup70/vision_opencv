import numpy as np

normal_list = []
file_path = "DepthCamera/xangle.txt"
flag = 0
with open(file_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("2026-01-10 08:50:07"):
            flag = 1
            continue
        if flag == 1 and line.startswith("Average normal vector"):
            flag = 0
            break
        if flag == 1:
            parts = line.strip().split(",")
            normal = [float(part) for part in parts]
            normal_list.append(normal)

all_points = len(normal_list)
normal_array = np.array(normal_list)
average = np.mean(normal_array, axis=0)
average /= np.linalg.norm(average)  # 单位化
average_good = np.array([0.0, 1.0, 0.0])
cos_thresh = np.cos(np.deg2rad(0.5))
times = [0,0]
while average_good @ average < cos_thresh:
    average = average_good
    dots = normal_array @ average
    k = int(0.7 * len(dots))               # 取最近 70%
    idx = np.argsort(dots)[:k]
    normal_array_good = normal_array[idx]   #先不删
    average_good = np.mean(normal_array_good, axis=0)
    average_good /= np.linalg.norm(average_good)
    times[0] += 1

average_good = np.array([0.0, 1.0, 0.0])
while average_good @ average < cos_thresh:
    average = average_good
    dots = normal_array @ average
    k = int(0.7 * len(dots))
    idx = np.argsort(dots)[:k]
    normal_array = normal_array[idx]   #删
    average_good = np.mean(normal_array, axis=0)
    average_good /= np.linalg.norm(average_good)
    times[1] += 1

print(f"Average normal vector: {average_good}\nPoints used: {len(normal_array)} in all points {all_points}\n")
print(f"Iterations without deletion: {times[0]}, with deletion: {times[1]}")