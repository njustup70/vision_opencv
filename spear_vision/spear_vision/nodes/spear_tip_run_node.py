"""
运行节点（只看大板）：输出 tip 位姿

特点：
- 与 spear_tip_node 共享同一套算法/参数/话题；
- 强制进入 run 模式，不进行外参标定；
- 适合“现场测量/对接”的长期运行场景。
"""

from __future__ import annotations

import rclpy
from rclpy.parameter import Parameter

from spear_vision.nodes.spear_tip_node import SpearTipNode


class SpearTipRunNode(SpearTipNode):
    def __init__(self) -> None:
        # 使用独立节点名，避免与标定节点冲突
        super().__init__(node_name="spear_tip_run")

        # 强制进入“运行”流程（只检测大板）
        self.set_parameters(
            [
                Parameter("mode", value="run"),
                # 运行模式不需要标定后自动退出
                Parameter("calib_auto_exit", value=False),
            ]
        )


def main() -> None:
    rclpy.init()
    node = SpearTipRunNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
