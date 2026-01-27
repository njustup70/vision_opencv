import cv2
import numpy as np
from deform_restore import ROIRestore
from light_preproc import clahe_on_v, preproc_hsv, get_red_blue_mask, pop_color
import matplotlib.pyplot as plt


def connect_axis_lines(edge_bin, kx=25, ky=25):
    # edge_bin: 0/255
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))  # 水平连线
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))  # 垂直连线

    h = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, hker)
    v = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, vker)

    return cv2.bitwise_or(h, v)

def keep_axis_aligned_long(edge_bin, min_length=80, ratio=5.0):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(edge_bin, connectivity=8)
    out = np.zeros_like(edge_bin)
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        length = max(w, h)
        aspect = max(w, h) / max(1, min(w, h))
        if length >= min_length and aspect >= ratio:
            out[labels == i] = 255
    return out

def fit_line_weighted_pca(lines_xyxy):
    L = np.asarray(lines_xyxy)

    # 关键：HoughLinesP 输出常是 (N,1,4)，先 squeeze 成 (N,4)
    if L.ndim == 3 and L.shape[1] == 1 and L.shape[2] == 4:
        L = L[:, 0, :]
    elif L.ndim == 2 and L.shape[1] == 4:
        pass
    elif L.ndim == 1 and L.shape[0] == 4:
        L = L.reshape(1, 4)
    else:
        raise ValueError(f"unexpected lines shape: {L.shape}")

    L = L.astype(np.float32)
    p1 = L[:, 0:2]
    p2 = L[:, 2:4]
    lens = np.linalg.norm(p2 - p1, axis=1) + 1e-6         # (K,)

    # 每条线段的两个端点，各自权重取该线段长度（也可取 lens/2，都一样比例）
    pts = np.vstack([p1, p2])                              # (2K,2)
    w   = np.hstack([lens, lens]).astype(np.float32)        # (2K,)

    # 加权均值
    w_sum = w.sum()
    mean = (pts * w[:, None]).sum(axis=0) / w_sum

    # 加权协方差（2x2）
    X = pts - mean
    C = (w[:, None] * X).T @ X / w_sum

    # 主方向：取最大特征值对应特征向量
    eigvals, eigvecs = np.linalg.eigh(C)
    d = eigvecs[:, np.argmax(eigvals)]                     # (2,)
    d = d / (np.linalg.norm(d) + 1e-6)

    # 用投影范围合成整条线段
    t = X @ d
    a = mean + d * t.min()
    b = mean + d * t.max()

    return [float(a[0]), float(a[1]), float(b[0]), float(b[1])], d, mean

# 根据参数ρ、θ画线
def draw_rho_theta(img, rho, theta, color=(0,0,255), thickness=2):
    """
    img: BGR image
    rho: 距离（像素）
    theta: 角度（弧度），注意是法向角
    """

    # 把 theta 规范到 [-0.25pi, 0.75pi)
    theta = (theta + 0.25*np.pi) % (np.pi) - 0.25*np.pi

    h, w = img.shape[:2]
    a = np.cos(theta)
    b = np.sin(theta)

    # 线上的一个点 (x0, y0)
    x0 = a * rho
    y0 = b * rho

    # 方向向量（沿着线走），与法向垂直
    dx = -b
    dy = a

    # 取足够长，保证穿过画面
    L = max(h, w) * 2
    x1 = int(round(x0 + dx * L))
    y1 = int(round(y0 + dy * L))
    x2 = int(round(x0 - dx * L))
    y2 = int(round(y0 - dy * L))

    cv2.line(img, (x1,y1), (x2,y2), color, thickness, cv2.LINE_AA)

# 找交点
def intersect_rho_theta(rho1, th1, rho2, th2, eps=1e-9):
        # 把 theta 规范到 [-0.25pi, 0.75pi)
    th1 = (th1 + 0.25*np.pi) % (np.pi) - 0.25*np.pi
    th2 = (th2 + 0.25*np.pi) % (np.pi) - 0.25*np.pi
    a1, b1 = np.cos(th1), np.sin(th1)
    a2, b2 = np.cos(th2), np.sin(th2)
    A = np.array([[a1, b1],
                  [a2, b2]], dtype=np.float64)
    det = np.linalg.det(A)
    if abs(det) < eps:
        return None  # 平行或近似平行
    rhs = np.array([rho1, rho2], dtype=np.float64)
    x, y = np.linalg.solve(A, rhs)
    return float(x), float(y)

# 聚类
def find_clusters(data, group_num=None, eps=0.02, min_samples=5, if_annular=False, data_range=None, neighbor=5):
    """
    给定一维数据，直方图聚类
    
    :param data: 一维数据数组
    :param group_num: 分的组数
    :param eps: 直方图bin宽度
    :param min_samples: 每组可采信最小样本数
    :param if_annular: 是否考虑环形数据
    :param data_range: 数据范围，仅在环形数据时使用，格式为 (min, max)
    :param neighbor: 合并峰时的邻近距离阈值
    :return: 聚类结果列表，每个元素为字典，包含'center'(中心值)，'count'(样本数)，'idx'(包含的数据下标列表)
    """ 

    depth_min, depth_max = np.min(data), np.max(data)
    if if_annular:
        if data_range is not None:
            depth_min, depth_max = data_range
        else:
            raise ValueError("For annular data, data_range must be provided.")
    bins = max(1, int((depth_max - depth_min) / eps)) # bin数量
    bins_idx = [[] for _ in range(bins)]
    bins_count = [0 for _ in range(bins)]
    bins_edges = np.linspace(depth_min, depth_max, bins+1)
    bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2  # 每个bin中心
    #hist, bin_edges = np.histogram(data, bins=bins, range=(depth_min, depth_max))
    for i in range(len(data)):
        d = data[i]
        bin_idx = int(((d - depth_min) / eps).item())
        if bin_idx == bins:
            bin_idx = bins - 1
        bins_idx[bin_idx].append(i)
        bins_count[bin_idx] += 1
        # print(d, "-> bin", bin_idx, "center", bins_centers[bin_idx])

    # 画柱状图
    # plt.bar(bins_centers, bins_count, width=eps*0.9)
    # plt.xlabel("Value")
    # plt.ylabel("Count")
    # plt.title("Histogram for Clustering")
    # plt.show()
    # plt.close()

    # 合并相邻的峰
    merged = [] # 大峰列表，元素为bin索引列表
    cur_group = [0]
    for i in range(1, bins):
        if bins_count[i] > 0:
            if i - cur_group[-1] <= neighbor:  # 相邻或近邻
                cur_group.append(i)
            else:
                merged.append(cur_group)
                cur_group = [i]
    merged.append(cur_group)
    
    if if_annular and len(merged) > 1:
        # 环形数据，检查首尾是否相连
        first = merged[0]
        last = merged[-1]
        wrap_gap = (first[0] + bins) - last[-1]

        if wrap_gap <= neighbor:
            # 合并首尾
            merged = merged[1:-1] + [last + first]
        # if merged[0][0] == 0 and merged[-1][-1] == bins - 1:
        #     new_group = merged[-1] + merged[0]
        #     merged = merged[1:-1]
        #     merged.append(new_group)

    # 计算每个大峰的加权平均中心、大小和包含的数据下标
    peaks_merged = []
    for group in merged:
        idx = [i for b in group for i in bins_idx[b]]
        total_count = sum([bins_count[b] for b in group])
        # print([bins_centers[b] for b in group], "counts:", [bins_count[b] for b in group])
        # center_depth = np.average([bins_centers[b] for b in group], weights=[bins_count[b] for b in group])
        if if_annular:
            # 环形数据，调整中心值到正确范围
            annular_center = (depth_min + depth_max) / 2.0
            bins_centers -= annular_center
            period = depth_max - depth_min
            v = np.asarray(bins_centers, np.float64)[group]
            w = np.asarray(bins_count, np.float64)[group]
            ang = 2*np.pi * (v / period)  # 映射到 [0, 2pi)
            c = np.sum(w * np.cos(ang))
            s = np.sum(w * np.sin(ang))
            mean_ang = np.arctan2(s, c)   # [-pi, pi]
            mean_deg = (mean_ang * period) / (2*np.pi)
            # 把结果规范到 [-period/2, period/2)
            mean_deg = (mean_deg + period/2) % period - period/2
            center_depth = mean_deg + annular_center
        else:
            center_depth = np.average([bins_centers[b] for b in group], weights=[bins_count[b] for b in group])

        peaks_merged.append({
            'center': center_depth,
            'count': int(total_count),
            'idx': idx
        })
        # idx = np.array(idx)
        # print("all data in this peak:", data[idx])
        # print(idx.min() , idx.max(), total_count, center_depth)
    # 按大小排序
    peaks_merged = sorted(peaks_merged, key=lambda x: x['count'], reverse=True)
    peaks_merged = [p for p in peaks_merged if p["count"] >= min_samples]
    if group_num is not None:
        peaks_merged = peaks_merged[:group_num]
    # 简单删一下离均值太远的点
    # for peak in peaks_merged:
    #     peak["idx"] = [i for i in peak["idx"] if abs(data[i] - peak["center"]) <= eps * len(peak["idx"])]
    return peaks_merged

def find_largest_quad(img_bgr):
    """
    在输入图像中寻找最大的矩形轮廓，返回其四个顶点的图像坐标

    :param img_bgr: 输入的BGR图像
    :return: 四个顶点的图像坐标，按顺时针顺序排列 :type:`np.ndarray` (4x2)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (11,11), 0)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=100, sigmaSpace=5)

    # 你也可以改成自适应阈值：cv2.adaptiveThreshold
    edges = cv2.Canny(gray, 30, 80)
    # cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
    # cv2.imshow("Canny Edges", edges)
    # cv2.waitKey(5000)

    # 闭运算让边更连贯
    #kernel_m = np.ones((7,7), np.uint8)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_m, iterations=1)
    edges = connect_axis_lines(edges, kx=10, ky=10)
    edges = keep_axis_aligned_long(edges, min_length=300, ratio=1.0)
    edges = connect_axis_lines(edges, kx=20, ky=20)
    # skel = cv2.ximgproc.thinning(edges // 255) * 255
    # edges = skel.astype(np.uint8)
    #edges = cv2.Canny(edges, 50, 150)
    # 膨胀
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel_d, iterations=1)

    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cv2.namedWindow("Processed Edges", cv2.WINDOW_NORMAL)
    cv2.imshow("Processed Edges", edges)
    cv2.waitKey(5000)

    lines = cv2.HoughLinesP(
        edges,
        rho=10,              # 5像素精度
        theta=np.pi/180*3,  # 3度步长
        threshold=60,      # 线段累计阈值,处于同一直线上的点数下限
        minLineLength=400,  # 过滤短线
        maxLineGap=25       # 线段间最大允许间隔
    )

    lines_img = img_bgr.copy()

    line_theta = np.degrees(np.arctan2(lines[:,:,3]-lines[:,:,1], lines[:,:,2]-lines[:,:,0]))
    # 无向化，theta = (theta + 0.5*pi) % pi - 0.5*pi 范围[-90,90)
    line_theta = (line_theta + 90) % 180 - 90
    # 对方向聚类
    theta_peaks = find_clusters(line_theta.reshape(-1), group_num=2, eps=1, min_samples=2, if_annular=True, data_range=(-90, 90), neighbor=3)
    # 计算法向角 = theta + 90
    # theta_normal = line_theta + 90
    # # r = x * np.cos(theta) + y * np.sin(theta)
    # line_r1 = lines[:,:,0] * np.cos(np.radians(theta_normal)) + lines[:,:,1] * np.sin(np.radians(theta_normal))
    # line_r2 = lines[:,:,2] * np.cos(np.radians(theta_normal)) + lines[:,:,3] * np.sin(np.radians(theta_normal))
    # line_r = (line_r1 + line_r2) / 2.0
    print([tp['center'] for tp in theta_peaks], "theta peaks found.")

    outer_frame = []
    if lines is not None:
        count = 0
        count_theta = 0
        color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,128,0), (0,128,255), (128,0,255), (128,255,0)]
        cv2.putText(lines_img, f"Total lines detected: {len(lines)}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        # print(len(theta_peaks), "theta peaks found.")
        for theta_peak in theta_peaks:
            #idx_theta_global = np.array(theta_peak['idx'], dtype=int)
            lines_average = []
            lines_theta_peak = lines[theta_peak['idx']]
            theta_ave = theta_peak['center']
            print("Processing theta peak at center {:.2f} with {} lines.".format(theta_ave, len(lines_theta_peak)))
            # 计算法向角 = theta + 90
            theta_normal_ave = theta_ave + 90
            # r = x * np.cos(theta) + y * np.sin(theta)
            line_r = (lines_theta_peak[:,:,0] + lines_theta_peak[:,:,2]) / 2 * np.cos(np.radians(theta_normal_ave)) + (lines_theta_peak[:,:,1] + lines_theta_peak[:,:,3]) / 2 * np.sin(np.radians(theta_normal_ave))
            r_peaks = find_clusters(line_r, group_num=None, eps=5, min_samples=1, if_annular=False, neighbor=5)
            # print(len(r_peaks), "r peaks found for theta peak at center {:.2f}.".format(theta_peak['center']))
            for r_peak in r_peaks:
                idx_r_global = np.array(theta_peak['idx'])[r_peak['idx']]
                line_peak = lines[idx_r_global]
                # print("Detected line peak - theta center: {:.2f}, r center: {:.2f}, count: {}".format(
                #     theta_peak['center'], r_peak['center'], len(line_peak)
                # ))
                # cv2.putText(lines_img, f"Lines detected in peak {count}: {len(line_peak)}", (30,30 + 40*(count+1)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_list[count % len(color_list)], 2)
                
                # for i in range(0, len(line_peak)):
                #     l = line_peak[i][0]
                #     cv2.line(lines_img, (l[0], l[1]), (l[2], l[3]), color_list[count % len(color_list)], 3, cv2.LINE_AA)
                average_line, direction, point_on_line = fit_line_weighted_pca(line_peak)
                cv2.line(lines_img, (int(average_line[0]), int(average_line[1])),
                         (int(average_line[2]), int(average_line[3])),
                         color_list[count % len(color_list)], 4, cv2.LINE_AA)
                line_ave_normal = np.array([-direction[1], direction[0]])
                line_ave_r = (average_line[0] + average_line[2]) / 2 * line_ave_normal[0] + (average_line[1] + average_line[3]) / 2 * line_ave_normal[1]
                # print("  Averaged line - points:{}, theta: {}, r: {:.2f}".format(
                #     average_line, line_ave_normal, line_ave_r
                # ))
                line_ave_r = abs(line_ave_r)
                lines_average.append([average_line, line_ave_r])
                count += 1
            # cv2.putText(lines_img, f"Lines detected in peak {count}: {len(lines_theta_peak)}", (30,30 + 40*(count+1)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_list[count % len(color_list)], 2)
            # for i in range(0, len(lines_theta_peak)):
            #     l = lines_theta_peak[i][0]
            #     cv2.line(lines_img, (l[0], l[1]), (l[2], l[3]), color_list[count % len(color_list)], 3, cv2.LINE_AA)
            # 寻找等间距线
            lines_average = sorted(lines_average, key=lambda x: x[1])  # 按 r 排序
            # lines_average = [[list(la[0]), list(la[1])]  for la in lines_average]  # 把元组变成列表，方便后续添加元素
            if len(lines_average) >= 2:
                rs = [lr[1] for lr in lines_average]
                # print(rs)
                delta_rs = [abs(rs[i] - rs[j]) for i in range(1, len(rs)) for j in range(i)]
                delta_rs.sort()
                print("Delta rs:", delta_rs)
                # 删0
                # delta_rs = [dr for dr in delta_rs if dr > 1e-3]
                # delta_rs.extend([dr/2 for dr in delta_rs])
                # delta_rs.extend([dr/3 for dr in delta_rs])
                delta_r_peaks = find_clusters(np.array(delta_rs), group_num=None, eps=15, min_samples=1, if_annular=False, neighbor=3)
                # 没招了，按范围筛吧
                for i, dr_peak in enumerate(delta_r_peaks):
                    if dr_peak['center'] >350 and dr_peak['center'] < 450:
                        dr_main_peak = dr_peak
                        break
                else:
                    delta_r_peaks.sort(key=lambda x: abs(x['center']-400))
                    dr_main_peak = delta_r_peaks[0]
                # plt.bar(np.arange(len(delta_rs)), delta_rs, width=0.8)
                # plt.xlabel("Index")
                # plt.ylabel("Delta R")
                # plt.title("Delta R between Lines")
                # plt.show()
                # plt.close()
                print("Found delta r peaks for theta at {:.2f}.".format(dr_main_peak['center'] if len(delta_r_peaks) > 0 else -1))
                if len(delta_r_peaks) > 0:
                    best_delta_r = dr_main_peak['center']
                    # 标记等间距线
                    tag = [0 for _ in range(len(lines_average))]
                    suit_lines = [[[] for _ in range(5)] for _ in range(len(lines_average))]
                    for i in range(len(lines_average)):
                        suit_lines[i][0] = lines_average[i][0]
                        # print(suit_lines[i][0])
                        suit_lines[i][0].append(lines_average[i][1])  # r 值
                        # if tag[i]:
                        #     continue
                        for j in range(i+1, len(lines_average)):
                            r_diff = abs(lines_average[j][1] - lines_average[i][1])
                            for k in range(1,5):
                                if abs(r_diff / k - best_delta_r) <= 30.0:
                                    l1 = lines_average[i][0]
                                    l2 = lines_average[j][0]
                                    suit_lines[i][k].append([l2, abs(r_diff / k - best_delta_r), lines_average[j][1]])
                                    tag[j] += 1
                                    print(i, j, "matched for k =", k, "r_diff =", r_diff, "is", lines_average[j][0])
                        # 选择误差最小的那个
                        
                        for k in range(1, 5):
                            if len(suit_lines[i][k]) > 0:
                                suit_lines[i][k].sort(key=lambda x: x[1])
                                print(suit_lines[i][k][0])
                                r = suit_lines[i][k][0][2]
                                suit_lines[i][k] = suit_lines[i][k][0][0].copy()  # 取误差最小的 l2
                                suit_lines[i][k].append(r)
                                print(i, "suit line k =", k, "is", suit_lines[i][k])
                            else:
                                suit_lines[i][k] = None

                    # 统计完毕，找出最佳组
                    # print(suit_lines)
                    max_count = 0
                    best_idx = -1
                    for i in range(len(suit_lines)):
                        count_nonnull = sum([1 for l in suit_lines[i] if l is not None])
                        print("Line group", i, "has", count_nonnull, "lines.")
                        if count_nonnull > max_count:
                            max_count = count_nonnull
                            best_idx = i
                            print("New best idx:", best_idx, "with count:", max_count)
                        elif count_nonnull == max_count and best_idx >=0:
                            # 比较 r 值方差，取小的
                            r_vals1 = [suit_lines[best_idx][k][4] for k in range(5) if suit_lines[best_idx][k] is not None]
                            r_vals2 = [suit_lines[i][k][4] for k in range(5) if suit_lines[i][k] is not None]
                            var1 = np.var(r_vals1) if len(r_vals1) >1 else 0
                            var2 = np.var(r_vals2) if len(r_vals2) >1 else 0
                            if var2 < var1:
                                best_idx = i
                            print("Tie in count:", count_nonnull, "comparing var", var1, "vs", var2)
                    print("the best line group has", max_count, "lines.")
                    # 绘制最佳组
                    line_outers = [] # 最外侧两条线
                    if best_idx >= 0:
                        l0 = suit_lines[best_idx][0]
                        r0 = suit_lines[best_idx][0][4]
                        k_max = max([k for k in range(5) if suit_lines[best_idx][k] is not None])
                        for k in range(k_max + 1):
                            if k == 0:
                                k_line_r = r0
                                k_line_theta = np.degrees(np.arctan2(l0[3]-l0[1], l0[2]-l0[0])) + 90
                                line_outers.append((k_line_r, k_line_theta))
                            draw_rho_theta(lines_img, r0 + k * best_delta_r, np.radians(theta_normal_ave), color=(0,0,255), thickness=2)
                            print(theta_normal_ave, r0 + k * best_delta_r)
                            l = suit_lines[best_idx][k]
                            if l is not None:
                                if k == k_max:
                                    line_outers.append((l[4], np.degrees(np.arctan2(l[3]-l[1], l[2]-l[0])) + 90))
                                cv2.line(lines_img, (int(l[0]), int(l[1])),
                                         (int(l[2]), int(l[3])),
                                         (0,0,0), 6, cv2.LINE_AA)
                                cv2.putText(lines_img, f"L{k+1}", (int((l[0]+l[2])/2), int((l[1]+l[3])/2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
                        # 补缺的
                        width = lines_img.shape[1]
                        height = lines_img.shape[0]
                        range_r = width if abs(np.cos(np.radians(theta_normal_ave))) > abs(np.sin(np.radians(theta_normal_ave))) else height

                        for i in range(1, k_max):
                            if suit_lines[best_idx][i] is None:
                                lines_average.sort(key=lambda x: abs(x[1] + (r0 + i * best_delta_r)))
                                l = lines_average[0][0]
                                delta_r = abs(lines_average[0][1] + (r0 + i * best_delta_r))
                                if delta_r >= 200:
                                    continue
                                cv2.line(lines_img, (int(l[0]), int(l[1])),
                                            (int(l[2]), int(l[3])),
                                            (0,0,0), 6, cv2.LINE_AA)
                                cv2.putText(lines_img, f"L{i+1}", (int((l[0]+l[2])/2), int((l[1]+l[3])/2)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

                        k = k_max
                        before_max = int(abs(r0 - -50) // best_delta_r)
                        after_max = int(abs(r0 + k * best_delta_r - (range_r)) // best_delta_r)
                        after_max -= (before_max + after_max - (5-k-1)) if (before_max + after_max - (5-k-1)) > 0 else 0
                        print("Filling missing line L", k+1, "before max:", before_max, "after max:", after_max)
                        for i in range(before_max):
                            draw_rho_theta(lines_img, r0 - (i+1) * best_delta_r, np.radians(theta_normal_ave), color=(255,0,0), thickness=2)
                            # 在原线组里找r最近的线段
                            lines_average.sort(key=lambda x: abs(x[1] - (r0 - (i+1) * best_delta_r)))
                            l = lines_average[0][0]
                            delta_r = abs(lines_average[0][1] - (r0 - (i+1) * best_delta_r))
                            if delta_r >= 200:
                                line_outers[0] = (r0 - (i+1) * best_delta_r, theta_normal_ave)
                                continue # 实在差太多就不画了
                            line_outers[0] = (l[4], np.degrees(np.arctan2(l[3]-l[1], l[2]-l[0])) + 90)
                            cv2.line(lines_img, (int(l[0]), int(l[1])),
                                        (int(l[2]), int(l[3])),
                                        (0,0,0), 6, cv2.LINE_AA)
                            cv2.putText(lines_img, f"L{k-i-1}", (int((l[0]+l[2])/2), int((l[1]+l[3])/2)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
                            
                        for i in range(after_max):
                            draw_rho_theta(lines_img, r0 + (k + i + 1) * best_delta_r, np.radians(theta_normal_ave), color=(255,0,0), thickness=2)
                            lines_average.sort(key=lambda x: abs(x[1] - (r0 + (k + i + 1) * best_delta_r)))
                            l = lines_average[0][0]
                            delta_r = abs(lines_average[0][1] - (r0 + (k + i + 1) * best_delta_r))
                            if delta_r >= 200:
                                line_outers[1] = (r0 + (k + i + 1) * best_delta_r, theta_normal_ave)
                                continue
                            line_outers[1] = (l[4], np.degrees(np.arctan2(l[3]-l[1], l[2]-l[0])) + 90)
                            cv2.line(lines_img, (int(l[0]), int(l[1])),
                                        (int(l[2]), int(l[3])),
                                        (0,0,0), 6, cv2.LINE_AA)
                            cv2.putText(lines_img, f"L{k+i+1}", (int((l[0]+l[2])/2), int((l[1]+l[3])/2)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
                            
                    print(len(line_outers))
                    if len(line_outers) == 2:
                        outer_frame.extend(line_outers)
            
            count_theta += 1
        for lr, theta in outer_frame:
            draw_rho_theta(lines_img, lr, np.radians(theta), color=(255,255,0), thickness=4)
            print("Outer frame line - theta: {}, r: {}".format(theta, lr))

        # 计算交点
        x1,y1 = intersect_rho_theta(outer_frame[0][0], np.radians(outer_frame[0][1]),
                                    outer_frame[2][0], np.radians(outer_frame[2][1]))
        x2,y2 = intersect_rho_theta(outer_frame[0][0], np.radians(outer_frame[0][1]),
                                    outer_frame[3][0], np.radians(outer_frame[3][1]))
        x3,y3 = intersect_rho_theta(outer_frame[1][0], np.radians(outer_frame[1][1]),
                                    outer_frame[2][0], np.radians(outer_frame[2][1]))
        x4,y4 = intersect_rho_theta(outer_frame[1][0], np.radians(outer_frame[1][1]),
                                    outer_frame[3][0], np.radians(outer_frame[3][1]))
        return np.array([[x1,y1],
                         [x2,y2],
                         [x4,y4],
                         [x3,y3]], dtype=np.float32)
        # cv2.circle(lines_img, (int(x1), int(y1)), 10, (0,255,0), -1)
        # cv2.circle(lines_img, (int(x2), int(y2)), 10, (0,255,0), -1)
        # cv2.circle(lines_img, (int(x3), int(y3)), 10, (0,255,0), -1)
        # cv2.circle(lines_img, (int(x4), int(y4)), 10, (0,255,0), -1)
        # cv2.namedWindow("Hough Lines", cv2.WINDOW_NORMAL)
        # cv2.imshow("Hough Lines", lines_img)
        # cv2.waitKey(5000)

    # for cnt in contours[:20]:
    #     area = cv2.contourArea(cnt)
    #     if area < 700:
    #         continue
    #     cnt2 = cv2.convexHull(cnt)
    #     peri = cv2.arcLength(cnt2, True)
    #     approx = cv2.approxPolyDP(cnt2, 0.04 * peri, True)
    #     if cv2.isContourConvex(approx):
    #         cnt_big = cnt   # 这个 area = 1.35e6 的轮廓

    #         rect = cv2.minAreaRect(cnt_big)   # center, (w,h), angle
    #         box  = cv2.boxPoints(rect)        # 4 points
    #         box  = box.astype(np.float32)
    #         quad = box.reshape(4, 2).astype(np.float32)
    #         print("Found quad with area:", approx)
    #         return quad
    return None


# def verify_3x3_grid(warped_bgr, debug=False):
#     g = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
#     g = cv2.GaussianBlur(g, (3,3), 0)

#     # 强化线：二值化 + 形态学提取线结构
#     bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                cv2.THRESH_BINARY_INV, 21, 5)

#     h, w = bw.shape
#     # 提取竖线
#     v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, w//25))
#     v = cv2.erode(bw, v_kernel, iterations=1)
#     v = cv2.dilate(v, v_kernel, iterations=2)

#     # 提取横线
#     h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h//25, 1))
#     hh = cv2.erode(bw, h_kernel, iterations=1)
#     hh = cv2.dilate(hh, h_kernel, iterations=2)

#     # 投影（统计每列/每行线像素数量）
#     proj_x = v.sum(axis=0) / 255.0
#     proj_y = hh.sum(axis=1) / 255.0

#     # 期望峰值位置在 1/3 和 2/3 附近
#     x1, x2 = w/3, 2*w/3
#     y1, y2 = h/3, 2*h/3

#     def peak_score(proj, t1, t2, tol=0.06):
#         # 在目标附近窗口内找最大值，和整体最大值比
#         n = len(proj)
#         win = int(tol * n)
#         a1 = int(max(0, t1 - win)); b1 = int(min(n, t1 + win))
#         a2 = int(max(0, t2 - win)); b2 = int(min(n, t2 + win))
#         pmax = max(proj.max(), 1e-6)
#         s1 = proj[a1:b1].max() / pmax
#         s2 = proj[a2:b2].max() / pmax
#         return s1, s2

#     sx1, sx2 = peak_score(proj_x, x1, x2)
#     sy1, sy2 = peak_score(proj_y, y1, y2)

#     ok = (sx1 > 0.6 and sx2 > 0.6 and sy1 > 0.6 and sy2 > 0.6)
#     if debug:
#         return ok, (sx1, sx2, sy1, sy2), bw, v, hh
#     return ok


# def get_3x3_cells(out_size=360):
#     # 返回 9 个小格的 (x0,y0,x1,y1)
#     cells = []
#     step = out_size / 3.0
#     for r in range(3):
#         for c in range(3):
#             x0 = int(round(c * step))
#             y0 = int(round(r * step))
#             x1 = int(round((c+1) * step))
#             y1 = int(round((r+1) * step))
#             cells.append((x0,y0,x1,y1))
#     return cells

def sharpen_unsharp(img, amount=0.8): # 非常简单的非锐化掩模锐化，放大高频
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    sharpened = cv2.addWeighted(
        img, 1 + amount,
        blur, -amount,
        0
    )
    return sharpened

def detect_3x3_outer_and_warp(img, out_size=360):


    quad = find_largest_quad(img)
    if quad is None:
        return None  # 没找到外框
    # else:
    #     print("Found quad:", quad)

    warped= ROIRestore(img, quad, image_shape=[out_size, out_size])

    # 删除下1/4部分，拉伸上3/4部分到全图
    h, w = warped.shape[:2]
    crop_h = int(h * 3 / 4)
    top = warped[:crop_h, :]

    stretched = cv2.resize(top, (w, h), interpolation=cv2.INTER_CUBIC)


    # cv2.namedWindow("Warped Preview", cv2.WINDOW_NORMAL)
    # cv2.imshow("Warped Preview", warped)
    # cv2.waitKey(5000)

    # ok = verify_3x3_grid(warped)
    # if not ok:
    #     return None  # 外框像四边形，但不是3x3网格

    # cells = get_3x3_cells(out_size)
    return {
        "quad": quad,       # 原图外框四点
        "warped": stretched,   # 拉正后的图
    }

def process_image_and_detect_3x3(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = preproc_hsv(hsv, v_gain=1.8, s_gain=1.4)
    hsv = clahe_on_v(hsv, clip=2.5, grid=(8,8))
    mask = get_red_blue_mask(hsv, blue=True, red=True, kernel_size=(7,7))
    hsv  = pop_color(hsv, mask, 1.4, 1.0) 
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 模糊
    # img = cv2.GaussianBlur(img, (3,3), 0)
    # 锐化
    #img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    img = sharpen_unsharp(img, amount=0.8)
    res = detect_3x3_outer_and_warp(img, out_size=360)
    # if res is None:
    #     print("No valid 3x3 grid found.")
    # else:
    #     print("3x3 grid detected.")
    #     warped = res["warped"]
    #     cells = res["cells"]

        # for (x0,y0,x1,y1) in cells:
        #     cv2.rectangle(warped, (x0,y0), (x1,y1), (0,255,0), 2)
    return res

def main():
    img_name = "1.jpg"
    img_path = "photos/" + img_name

    img = cv2.imread(img_path)
    img_original = img.copy()

    process_image_and_detect_3x3(img)

    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img_original)
    # cv2.imshow("Warped with Cells", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()