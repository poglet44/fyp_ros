from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    slam_launch = os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')

    pointcloud_to_scan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        parameters=[{
            'target_frame': 'base_scan',
            'transform_tolerance': 0.1,
            'min_height': 0.0,
            'max_height': 0.3,
            'angle_min': -3.14159,
            'angle_max': 3.14159,
            'angle_increment': 0.0087,
            'scan_time': 0.1,
            'range_min': 0.3,
            'range_max': 12.0,
            'use_inf': True,
            'inf_epsilon': 1.0,
        }],
        remappings=[
            ('cloud_in', '/points'),
            ('scan', '/scan'),
        ],
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch),
        launch_arguments={
            'use_sim_time': 'true',
        }.items()
    )

    return LaunchDescription([
        pointcloud_to_scan,
        slam_toolbox,
    ])