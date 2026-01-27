"""
ROS 消息转换工具（与 core 分离，便于 core 无 ROS 依赖）
"""

from __future__ import annotations

from geometry_msgs.msg import PoseStamped, TransformStamped

from spear_vision.utils.tf_utils import Rt, matrix_to_quaternion, rodrigues_to_matrix


def rt_to_pose_stamped(rt: Rt, stamp, frame_id: str) -> PoseStamped:
    # Rt -> PoseStamped
    pose = PoseStamped()
    pose.header.stamp = stamp
    pose.header.frame_id = frame_id
    qx, qy, qz, qw = matrix_to_quaternion(rodrigues_to_matrix(rt.rvec))
    pose.pose.position.x = float(rt.tvec[0])
    pose.pose.position.y = float(rt.tvec[1])
    pose.pose.position.z = float(rt.tvec[2])
    pose.pose.orientation.x = float(qx)
    pose.pose.orientation.y = float(qy)
    pose.pose.orientation.z = float(qz)
    pose.pose.orientation.w = float(qw)
    return pose


def rt_to_transform_stamped(rt: Rt, stamp, parent_frame: str, child_frame: str) -> TransformStamped:
    # Rt -> TransformStamped
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    qx, qy, qz, qw = matrix_to_quaternion(rodrigues_to_matrix(rt.rvec))
    t.transform.translation.x = float(rt.tvec[0])
    t.transform.translation.y = float(rt.tvec[1])
    t.transform.translation.z = float(rt.tvec[2])
    t.transform.rotation.x = float(qx)
    t.transform.rotation.y = float(qy)
    t.transform.rotation.z = float(qz)
    t.transform.rotation.w = float(qw)
    return t
