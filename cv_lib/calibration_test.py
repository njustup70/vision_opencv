import os
import cv2
import numpy as np
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
z1 = (400-75)/1000
z2 = (2020-75)/1000

world_points = np.array([
    [x1, y1, z1],
    [x2, y1, z1],
    [x2, y1, z2],
    [x1, y1, z2],
], dtype=np.float64)

T_cam_base = [[ 0.05309445, -0.99672328, -0.06102193, -0.06469527],
 [ 0.41525076,  0.0776115,  -0.90639024, -0.41730752],
 [ 0.90815626,  0.02278489,  0.41801084, -0.11940187],
 [ 0.,          0.,          0.,          1.        ]]

def quat_to_R(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)

    x, y, z, w = q

    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,     1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w,     1 - 2 * x * x - 2 * y * y]
    ], dtype=np.float64)

    return R


def transform_stamped_to_matrix(tf):
    """
    TransformStamped -> T_parent_child

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


def transform_points(T, points):
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points, ones])
    points_out_h = (T @ points_h.T).T
    return points_out_h[:, :3]


class SimpleTFBuffer:
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
    img = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding == "bgr8":
        img = img.reshape(msg.height, msg.width, 3)
        return img.copy()

    if msg.encoding == "rgb8":
        img = img.reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if msg.encoding == "mono8":
        img = img.reshape(msg.height, msg.width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if msg.encoding == "bgra8":
        img = img.reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if msg.encoding == "rgba8":
        img = img.reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    raise ValueError(f"暂不支持的图像编码: {msg.encoding}")


def project_map_points_to_image(
    map_points,
    T_map_base,
    T_cam_base,
    K,
    image_shape,
    dist=None
):
    """
    map_points: Nx3，map 坐标系下的 3D 点
    T_map_base: 4x4
    T_cam_base: 4x4
    K: 3x3
    image_shape: img.shape

    返回:
        projected_points: [(u, v, point_index), ...]
    """

    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)

    h, w = image_shape[:2]

    T_base_map = invert_T(T_map_base)

    # map -> base
    base_points = transform_points(T_base_map, map_points)

    # base -> camera
    cam_points = transform_points(T_cam_base, base_points)

    projected_points = []

    for i, p_cam in enumerate(cam_points):
        X, Y, Z = p_cam

        # 在相机后方，忽略
        if Z <= 1e-6:
            continue

        # 手动投影，等价于 K @ [X/Z, Y/Z, 1]
        u = K[0, 0] * X / Z + K[0, 2]
        v = K[1, 1] * Y / Z + K[1, 2]

        if not np.isfinite(u) or not np.isfinite(v):
            continue

        # 不在图像范围内，忽略
        if u < 0 or u >= w or v < 0 or v >= h:
            continue

        projected_points.append((int(round(u)), int(round(v)), i))

    return projected_points


def order_quad_points(points):
    """
    将四个点排序为:
    左上、右上、右下、左下
    """
    pts = np.array(points, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def draw_projected_points(img, projected_points):
    vis = img.copy()

    # 只取 u, v
    corners = [(u, v) for u, v, idx in projected_points]

    for u, v, idx in projected_points:
        u, v = int(round(u)), int(round(v))
        cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(
            vis,
            str(idx),
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    if len(corners) != 4:
        return vis

    # 排序：左上、右上、右下、左下
    tl, tr, br, bl = order_quad_points(corners)

    # 转 int 方便 cv2 画线
    def to_int_pt(p):
        return tuple(np.round(p).astype(int))

    # 先画外边框
    quad = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(vis, [quad], isClosed=True, color=(0, 0, 255), thickness=2)

    # 横竖各 4 条均分线
    # 如果是 4 条内部线，就需要分成 5 等份，所以 t = 1/5, 2/5, 3/5, 4/5
    for i in range(1, 3):
        t = i / 3.0

        # 竖线：上边 tl->tr 和下边 bl->br 对应点相连
        top = tl * (1 - t) + tr * t
        bottom = bl * (1 - t) + br * t
        cv2.line(vis, to_int_pt(top), to_int_pt(bottom), (0, 0, 255), 2)

        # 横线：左边 tl->bl 和右边 tr->br 对应点相连
        left = tl * (1 - t) + bl * t
        right = tr * (1 - t) + br * t
        cv2.line(vis, to_int_pt(left), to_int_pt(right), (0, 0, 255), 2)

    return vis


def load_tf_static_to_buffer(bag_path, storage_id="mcap"):
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


def test_project_all_images_from_mcap(
    bag_path,
    map_points,
    T_cam_base,
    K,
    image_topic="/camera/color/image_raw",
    target_frame="map",
    source_frame="base_link",
    storage_id="mcap",
    save_dir=None,
    save_video_path=None,
    playback_rate=4.0,
    show=True,
    max_frames=None
):
    """
    遍历 mcap 包里的全部图像帧。
    每一帧用当前最新 TF 查询 T_map_base_link，
    然后把固定 map 坐标下的 3D 点投影到图像上。

    map_points:
        Nx3，单位必须和 T_map_base 的平移单位一致，通常是 m。

    T_cam_base:
        4x4，solvePnP 算出来的外参。
        含义:
            p_cam = T_cam_base @ p_base
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

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
    for topic_name, type_name in type_map.items():
        print(f"  {topic_name}: {type_name}")

    required_topics = [image_topic, "/tf", "/tf_static"]
    existing_topics = [t for t in required_topics if t in type_map]

    reader.set_filter(rb2.StorageFilter(topics=existing_topics))

    msg_type_cache = {}

    def get_msg_type(topic):
        if topic not in msg_type_cache:
            msg_type_cache[topic] = get_message(type_map[topic])
        return msg_type_cache[topic]

    if show:
        cv2.namedWindow("projection test", cv2.WINDOW_NORMAL)

    video_writer = None
    last_image_timestamp = None
    frame_count = 0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        msg_type = get_msg_type(topic)
        msg = deserialize_message(data, msg_type)

        if topic == "/tf" or topic == "/tf_static":
            tf_buffer.update_tf_message(msg, timestamp)
            continue

        if topic != image_topic:
            continue

        img = ros_image_to_cv2(msg)

        try:
            T_map_base = tf_buffer.lookup(
                target_frame=target_frame,
                source_frame=source_frame
            )
            T_map_base[2, 3] = H_zone3 
        except RuntimeError as e:
            print(f"[skip] frame {frame_count}, timestamp={timestamp}: {e}")
            continue

        projected_points = project_map_points_to_image(
            map_points=map_points,
            T_map_base=T_map_base,
            T_cam_base=T_cam_base,
            K=K,
            image_shape=img.shape
        )

        vis = draw_projected_points(img, projected_points)

        # 左上角打印信息
        cv2.putText(
            vis,
            f"frame={frame_count}, ts={timestamp}, visible={len(projected_points)}/{len(map_points)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"proj_{frame_count:06d}.jpg")
            cv2.imwrite(save_path, vis)

        if save_video_path is not None:
            if video_writer is None:
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    save_video_path,
                    fourcc,
                    30.0,
                    (w, h)
                )

            video_writer.write(vis)

        if show:
            if last_image_timestamp is not None:
                dt_ns = timestamp - last_image_timestamp
                wait_ms = int(max(1, dt_ns / 1e6 / playback_rate))
            else:
                wait_ms = 1

            cv2.imshow("projection test", vis)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break

            last_image_timestamp = timestamp

        print(
            f"frame={frame_count}, timestamp={timestamp}, "
            f"visible={len(projected_points)}/{len(map_points)}"
        )

        frame_count += 1

        if max_frames is not None and frame_count >= max_frames:
            break

    if video_writer is not None:
        video_writer.release()

    if show:
        cv2.destroyWindow("projection test")

    print(f"done, processed image frames: {frame_count}")



test_project_all_images_from_mcap(
    bag_path= bag_path,
    map_points=world_points,
    T_cam_base=T_cam_base,
    K=K,
    image_topic="/camera/color/image_raw",
    target_frame="map",
    source_frame="base_link",
    storage_id="mcap",
    save_dir=None,
    save_video_path=None,
    playback_rate=4.0,
    show=True
)