from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare("fyp_exploration"),
        "config",
        "exploration_experiment_logger.yaml",
    ])

    return LaunchDescription([
        Node(
            package="fyp_exploration",
            executable="exploration_experiment_logger",
            name="exploration_experiment_logger",
            output="screen",
            parameters=[config_file],
        )
    ])
