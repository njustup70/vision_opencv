"""
spear_vision：面向精密装配/对接场景的视觉工具包（ROS 2）

主要能力：
- ChArUco 相机内参标定（导出 YAML）
- 单板 ChArUco/ArUco Board 位姿估计（PnP + 门控 + TF）
- 双板（大板 + 小 5x3）联合估计与 tip 位姿输出（用于矛头/矛杆对接）
"""
