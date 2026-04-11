from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument('node_type', default_value='R1', description='qr_detect 节点模式: R1 或 R2'))

    # camera parameters
    ld.add_action(DeclareLaunchArgument('camera_index', default_value='0', description='摄像头索引'))
    ld.add_action(DeclareLaunchArgument('fps', default_value='60', description='摄像头帧率'))
    ld.add_action(DeclareLaunchArgument('brightness', default_value='10.0', description='摄像头亮度'))
    ld.add_action(DeclareLaunchArgument('contrast', default_value='8.0', description='摄像头对比度'))
    ld.add_action(DeclareLaunchArgument('exposure', default_value='100.0', description='摄像头曝光'))

    # R1
    ld.add_action(DeclareLaunchArgument('qr_size_cm', default_value='15', description='生成二维码目标尺寸（cm）'))
    ld.add_action(DeclareLaunchArgument('qr_dpi', default_value='220', description='生成二维码 DPI'))
    ld.add_action(DeclareLaunchArgument('qr_save_dir', default_value='qr_codes', description='二维码保存路径（相对于节点工作目录）'))
    ld.add_action(DeclareLaunchArgument('screen_x', default_value='2560', description='显示屏 X 偏移'))
    ld.add_action(DeclareLaunchArgument('screen_y', default_value='0', description='显示屏 Y 偏移'))
    ld.add_action(DeclareLaunchArgument('screen_width', default_value='2160', description='显示屏 宽'))
    ld.add_action(DeclareLaunchArgument('screen_height', default_value='1440', description='显示屏 高'))

    # Nodes
    camera_node = Node(
        package='qr_detect',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'camera_index': LaunchConfiguration('camera_index'),
            'fps': LaunchConfiguration('fps'),
            'brightness': LaunchConfiguration('brightness'),
            'contrast': LaunchConfiguration('contrast'),
            'exposure': LaunchConfiguration('exposure')
        }]
    )

    qr_node = Node(
        package='qr_detect',
        executable='qr_detect_node',
        name='qr_detect_node',
        output='screen',
        parameters=[{
            'node_type': LaunchConfiguration('node_type'),
            'qr_size_cm': LaunchConfiguration('qr_size_cm'),
            'qr_dpi': LaunchConfiguration('qr_dpi'),
            'qr_save_dir': LaunchConfiguration('qr_save_dir'),
            'screen_x': LaunchConfiguration('screen_x'),
            'screen_y': LaunchConfiguration('screen_y'),
            'screen_width': LaunchConfiguration('screen_width'),
            'screen_height': LaunchConfiguration('screen_height')
        }]
    )

    ld.add_action(camera_node)
    ld.add_action(qr_node)

    return ld
