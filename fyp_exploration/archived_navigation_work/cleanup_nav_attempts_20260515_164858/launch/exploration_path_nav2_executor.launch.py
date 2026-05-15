from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    default_params_file = PathJoinSubstitution([
        FindPackageShare("fyp_exploration"),
        "config",
        "exploration_path_nav2_executor.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Path-based Nav2 executor parameter file.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time.",
        ),

        Node(
            package="fyp_exploration",
            executable="exploration_path_nav2_executor",
            name="exploration_path_nav2_executor",
            output="screen",
            parameters=[
                params_file,
                {"use_sim_time": use_sim_time},
            ],
        ),
    ])
