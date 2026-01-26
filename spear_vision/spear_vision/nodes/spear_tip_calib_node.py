"""
外参标定节点（大板 + 小板）：用于固化 primary_T_secondary

特点：
- 与 spear_tip_node 共享同一套算法/参数/话题；
- 强制进入 calibrate 模式，并在标定完成后自动退出；
- 适合“标定外参时一键启动，标定完成后自动结束”的流程。
"""

from __future__ import annotations

import rclpy
from rclpy.parameter import Parameter

from spear_vision.nodes.spear_tip_node import SpearTipNode


class SpearTipCalibNode(SpearTipNode):
    def __init__(self) -> None:
        # 使用独立节点名，避免与运行节点冲突
        super().__init__(node_name="spear_tip_calib")

        # 强制进入“外参标定”流程
        self.set_parameters(
            [
                Parameter("mode", value="calibrate"),
                # 标定完成后自动退出（默认 True，允许用户手动关闭）
                Parameter("calib_auto_exit", value=bool(self.get_parameter("calib_auto_exit").value)),
            ]
        )


def main() -> None:
    rclpy.init()
    node = SpearTipCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
