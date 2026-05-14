from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare('fyp_exploration'),
        'config',
        'frontier_detector.yaml',
    ])

    return LaunchDescription([
        Node(
            package='fyp_exploration',
            executable='frontier_detector',
            name='frontier_detector',
            output='screen',
            parameters=[config_file],
        )
    ])
