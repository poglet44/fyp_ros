from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare("fyp_exploration"),
        "config",
        "experiment_bag_recorder.yaml",
    ])

    return LaunchDescription([
        Node(
            package="fyp_exploration",
            executable="experiment_bag_recorder",
            name="experiment_bag_recorder",
            output="screen",
            parameters=[config_file],
        )
    ])
