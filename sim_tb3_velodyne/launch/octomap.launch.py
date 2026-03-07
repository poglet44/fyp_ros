from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('sim_tb3_velodyne')
    params_file = os.path.join(pkg_share, 'config', 'octomap_params.yaml')

    return LaunchDescription([
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': True}],
            remappings=[('cloud_in', '/points')],
        )
    ])