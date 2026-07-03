import time
import numpy as np
import cv2
import rosbag2_py as rb2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from collections import deque

frame_time_ns = [
    1783072577_160_949_110,
    1783072579_565_651_088,
    1783072585_744_075_075,
    1783072592_273_826_731
]

H_zone3 = (400+350)/1000
bag_path = "/home/Elaina/yolo/07-03-09-55"

# 相机内参
K = np.array([
    [611.544189453125, 0.0, 636.7078857421875],
    [0.0, 611.3428955078125, 397.55560302734375],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

x1 = (12100-100-440-1620)/1000
x2 = (12100-100-440)/1000
y1 = (6000-125)/1000
z1 = 400/1000
z2 = 2020/1000

# 三区框四角的3d点坐标，红区场地
world_points = np.array([
    [x1, y1, z1],
    [x2, y1, z1],
    [x2, y1, z2],
    [x1, y1, z2],
], dtype=np.float64)

def quat_to_R(qx, qy, qz, qw):
    """
    ROS 四元数 x,y,z,w -> 3x3 旋转矩阵
    不依赖 scipy
    """
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)

    x, y, z, w = q

    R = np.array([
        [1 - 2 * y * y - 2 * z * z,     2 * x * y - 2 * z * w,         2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,         1 - 2 * x * x - 2 * z * z,     2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,         2 * y * z + 2 * x * w,         1 - 2 * x * x - 2 * y * y]
    ], dtype=np.float64)

    return R


def transform_stamped_to_matrix(tf):
    """
    TransformStamped -> T_parent_child

    含义:
        p_parent = T_parent_child @ p_child
    """
    t = tf.transform.translation
    q = tf.transform.rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]

    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


class SimpleTFBuffer:
    """
    维护最新 TF。

    存储:
        parent -> child : T_parent_child

    查询:
        lookup("map", "base_link") 返回 T_map_base_link
    """

    def __init__(self):
        self.transforms = {}

    def update_tf_message(self, tf_msg, timestamp_ns=None):
        for tf in tf_msg.transforms:
            parent = tf.header.frame_id.strip("/")
            child = tf.child_frame_id.strip("/")

            if parent == "" or child == "":
                continue

            T_parent_child = transform_stamped_to_matrix(tf)

            self.transforms[(parent, child)] = {
                "T": T_parent_child,
                "timestamp": timestamp_ns
            }

    def lookup(self, target_frame, source_frame):
        """
        返回 T_target_source

        含义:
            p_target = T_target_source @ p_source
        """
        target_frame = target_frame.strip("/")
        source_frame = source_frame.strip("/")

        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)

        graph = {}

        for (parent, child), item in self.transforms.items():
            T_parent_child = item["T"]
            T_child_parent = invert_T(T_parent_child)

            graph.setdefault(parent, []).append((child, T_parent_child))
            graph.setdefault(child, []).append((parent, T_child_parent))

        queue = deque()
        queue.append((target_frame, np.eye(4, dtype=np.float64)))

        visited = set()

        while queue:
            current_frame, T_target_current = queue.popleft()

            if current_frame in visited:
                continue

            visited.add(current_frame)

            for next_frame, T_current_next in graph.get(current_frame, []):
                if next_frame in visited:
                    continue

                T_target_next = T_target_current @ T_current_next

                if next_frame == source_frame:
                    return T_target_next

                queue.append((next_frame, T_target_next))

        raise RuntimeError(f"找不到 TF 链: {target_frame} <- {source_frame}")


def ros_image_to_cv2(msg):
    """
    sensor_msgs/msg/Image -> OpenCV BGR 图像
    """
    img = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding in ["bgr8", "rgb8"]:
        img = img.reshape(msg.height, msg.width, 3)

        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img.copy()

    elif msg.encoding == "mono8":
        img = img.reshape(msg.height, msg.width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif msg.encoding in ["bgra8", "rgba8"]:
        img = img.reshape(msg.height, msg.width, 4)

        if msg.encoding == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img.copy()

    else:
        raise ValueError(f"暂不支持的图像编码: {msg.encoding}")


def load_tf_static_to_buffer(bag_path, storage_id="mcap"):
    """
    先完整扫一遍 /tf_static。
    因为 /tf_static 可能在 bag 开头，只从 start_time seek 可能会漏掉。
    """
    tf_buffer = SimpleTFBuffer()

    storage_options = rb2.StorageOptions(
        uri=bag_path,
        storage_id=storage_id
    )

    converter_options = rb2.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rb2.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    if "/tf_static" not in type_map:
        return tf_buffer

    reader.set_filter(rb2.StorageFilter(topics=["/tf_static"]))

    tf_msg_type = get_message(type_map["/tf_static"])

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic != "/tf_static":
            continue

        msg = deserialize_message(data, tf_msg_type)
        tf_buffer.update_tf_message(msg, timestamp)

    return tf_buffer


def read_play_region_and_capture(
    bag_path,
    start_time_ns,
    end_time_ns,
    target_time_ns,
    image_topic="/camera/color/image_raw",
    target_frame="map",
    source_frame="base_link",
    playback_rate=2.0,
    max_dt_ns=50_000_000,
    storage_id="mcap",
    show_image=True
):
    """
    在固定时间段内倍速播放 bag，并在指定时刻获取:
        1. 最近图像帧
        2. T_target_source，例如 T_map_base_link

    返回:
        frame_data = {
            "image": [item or None, ...],
            "T_target_source": [4x4 or None, ...],
            "tf_dt_ms": [...],
            "image_dt_ms": [...],
        }
    """

    target_time_ns = list(target_time_ns)
    target_count = len(target_time_ns)

    targets = sorted(
        [(t, i) for i, t in enumerate(target_time_ns)],
        key=lambda x: x[0]
    )

    frame_data = {
        "image": [None] * target_count,
        "T_target_source": [None] * target_count,
        "tf_dt_ms": [None] * target_count,
        "image_dt_ms": [None] * target_count,
    }

    # 先加载静态 TF
    tf_buffer = load_tf_static_to_buffer(
        bag_path=bag_path,
        storage_id=storage_id
    )

    storage_options = rb2.StorageOptions(
        uri=bag_path,
        storage_id=storage_id
    )

    converter_options = rb2.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rb2.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    print("type_map:")
    for k, v in type_map.items():
        print(f"  {k}: {v}")

    required_topics = [image_topic, "/tf", "/tf_static"]
    existing_topics = [t for t in required_topics if t in type_map]

    reader.set_filter(rb2.StorageFilter(topics=existing_topics))

    try:
        reader.seek(start_time_ns)
    except Exception as e:
        print(f"reader.seek({start_time_ns}) 失败，将从头开始读: {e}")

    msg_type_cache = {}

    def get_msg_type(topic):
        if topic not in msg_type_cache:
            msg_type_cache[topic] = get_message(type_map[topic])
        return msg_type_cache[topic]

    if show_image:
        cv2.namedWindow("bag playback", cv2.WINDOW_NORMAL)

    last_image_timestamp = None
    left = 0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if timestamp < start_time_ns:
            continue

        if timestamp > end_time_ns:
            break

        # 丢掉已经超过窗口的目标
        while left < len(targets) and timestamp > targets[left][0] + max_dt_ns:
            left += 1

        msg_type = get_msg_type(topic)
        msg = deserialize_message(data, msg_type)

        # 更新 TF buffer
        if topic == "/tf" or topic == "/tf_static":
            tf_buffer.update_tf_message(msg, timestamp)

            # 在目标时间附近尝试记录 T_target_source
            j = left
            while j < len(targets):
                target_t, original_i = targets[j]
                dt = timestamp - target_t

                if dt < -max_dt_ns:
                    j += 1
                    continue

                if dt > max_dt_ns:
                    break

                old_dt_ms = frame_data["tf_dt_ms"][original_i]
                if old_dt_ms is None or abs(dt / 1e6) < abs(old_dt_ms):
                    try:
                        T_target_source = tf_buffer.lookup(
                            target_frame=target_frame,
                            source_frame=source_frame
                        )
                        T_target_source[2, 3] = H_zone3 # 固定三区高于一区高度

                        frame_data["T_target_source"][original_i] = T_target_source
                        frame_data["tf_dt_ms"][original_i] = dt / 1e6

                    except RuntimeError:
                        pass

                j += 1

        # 播放图像，并记录目标时刻附近图像
        elif topic == image_topic:
            img_bgr = ros_image_to_cv2(msg)

            if show_image:
                if last_image_timestamp is not None:
                    delta_ns = timestamp - last_image_timestamp
                    wait_ms = int(max(1, delta_ns / 1e6 / playback_rate))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord("q"):
                        break

                cv2.imshow("bag playback", img_bgr)
                last_image_timestamp = timestamp

            j = left
            while j < len(targets):
                target_t, original_i = targets[j]
                dt = timestamp - target_t

                if dt < -max_dt_ns:
                    j += 1
                    continue

                if dt > max_dt_ns:
                    break

                old_item = frame_data["image"][original_i]
                if old_item is None or abs(dt) < abs(old_item["dt"]):
                    frame_data["image"][original_i] = {
                        "target_time": target_time_ns[original_i],
                        "timestamp": timestamp,
                        "dt": dt,
                        "dt_ms": dt / 1e6,
                        "msg": msg,
                        "img_bgr": img_bgr.copy(),
                        "type": type_map[topic],
                    }
                    frame_data["image_dt_ms"][original_i] = dt / 1e6

                j += 1

    if show_image:
        cv2.destroyWindow("bag playback")

    return frame_data



# 固定播放区域，建议比目标帧前后多留一点时间
start_time_ns = min(frame_time_ns) - 1_000_000_000
end_time_ns = max(frame_time_ns) + 1_000_000_000

frame_data = read_play_region_and_capture(
    bag_path=bag_path,
    start_time_ns=start_time_ns,
    end_time_ns=end_time_ns,
    target_time_ns=frame_time_ns,
    image_topic="/camera/color/image_raw",
    target_frame="map",
    source_frame="base_link",
    playback_rate=4.0,
    max_dt_ns=50_000_000,
    storage_id="mcap",
    show_image=True
)

for i, target_t in enumerate(frame_time_ns):
    print("=" * 80)
    print("frame index:", i)
    print("target time:", target_t)

    img_item = frame_data["image"][i]
    T_map_base = frame_data["T_target_source"][i]

    if img_item is None:
        print("image: 没有找到附近图像")
    else:
        print("image timestamp:", img_item["timestamp"])
        print("image dt_ms:", img_item["dt_ms"])

    if T_map_base is None:
        print("T_map_base_link: 没有找到")
    else:
        print("tf dt_ms:", frame_data["tf_dt_ms"][i])
        print("T_map_base_link:")
        print(T_map_base)

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T

def transform_points(T, points):
    """
    T: 4x4
    points: Nx3
    return: Nx3
    """
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)

    points_h = np.hstack([points, ones])
    points_transformed_h = (T @ points_h.T).T

    return points_transformed_h[:, :3]

def solve_multi_frame_direct(frames, K, dist=None):
    """
    直接把多帧的世界点转到各自的车体坐标系，
    然后一次性 solvePnP 求 T_cam_base。

    frames:
        [
            {
                "world_points": Nx3,
                "image_points": Nx2,
                "T_world_base": 4x4
            },
            ...
        ]

    返回：
        T_cam_base
    """

    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(dist, dtype=np.float64)

    all_base_points = []
    all_image_points = []

    for frame in frames:
        world_points = np.asarray(frame["world_points"], dtype=np.float64)
        image_points = np.asarray(frame["image_points"], dtype=np.float64)
        T_world_base = np.asarray(frame["T_world_base"], dtype=np.float64)

        T_base_world = invert_T(T_world_base)

        # 把世界点转换到当前帧车体坐标系下
        base_points = transform_points(T_base_world, world_points)

        all_base_points.append(base_points)
        all_image_points.append(image_points)

    all_base_points = np.vstack(all_base_points).astype(np.float64)
    all_image_points = np.vstack(all_image_points).astype(np.float64)

    # 多帧合起来点数通常 > 4，不建议继续用 IPPE
    # 用 ITERATIVE 或 EPNP 更合适
    success, rvec, tvec = cv2.solvePnP(
        all_base_points,
        all_image_points,
        np.asarray(K, dtype=np.float64),
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("multi-frame solvePnP failed")

    R_cam_base, _ = cv2.Rodrigues(rvec)
    T_cam_base = make_T(R_cam_base, tvec)

    return T_cam_base

# 鼠标取点
window_name = "select 4 points"
def draw_points(img, points):
    """
    在图上画已经点击的点
    """
    vis = img.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            vis,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return vis


def mouse_callback(event, x, y, flags, param):
    current_points = []

    img = param["img"]
    current_points = param["points"]

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 4:
            current_points.append((x, y))
            print(f"point {len(current_points)} = ({x}, {y})")

        vis = draw_points(img, current_points)
        cv2.imshow(window_name, vis)

    return current_points

def select_points_for_images(images):

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    all_points = []

    for img in images:
        current_points = []
        cv2.setMouseCallback(
            window_name,
            mouse_callback,
            {
                "img": img,
                "points": current_points
            }
        )

        cv2.imshow(window_name, img)

        print("=" * 60)
        print("请用鼠标左键点击 4 个点")

        while True:
            key = cv2.waitKey(20) & 0xFF

            # 点满 4 个后自动保存并进入下一张
            if len(current_points) == 4:
                all_points.append(current_points.copy())
                print(f"已保存 4 个点: {current_points}")
                break

    cv2.destroyAllWindows()
    return all_points

images = []
valid_indices = []

for i, item in enumerate(frame_data["image"]):
    if item is None:
        continue

    images.append(item["img_bgr"])
    valid_indices.append(i)

points_result = select_points_for_images(images)

frame_data["2d_points"] = [None] * len(frame_time_ns)

for idx, points in zip(valid_indices, points_result):
    frame_data["2d_points"][idx] = points

frames = []

for i in range(len(frame_time_ns)):
    if frame_data["2d_points"][i] is None:
        print(f"frame {i}: 没有 2D 点，跳过")
        continue

    if frame_data["T_target_source"][i] is None:
        print(f"frame {i}: 没有 T_map_base_link，跳过")
        continue

    image_points = np.array(frame_data["2d_points"][i], dtype=np.float64)

    T_map_base_link = frame_data["T_target_source"][i]

    frames.append({
        "world_points": world_points,
        "image_points": image_points,
        "T_world_base": T_map_base_link
    })

T_cam_base = solve_multi_frame_direct(
    frames,
    K
)

print("T_cam_base:")
print(T_cam_base)

T_base_cam = invert_T(T_cam_base)

print("T_base_cam:")
print(T_base_cam)