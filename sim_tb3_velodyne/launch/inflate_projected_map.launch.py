from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    inflation_radius_m = LaunchConfiguration("inflation_radius_m")
    occupied_threshold = LaunchConfiguration("occupied_threshold")
    unknown_border_padding_m = LaunchConfiguration("unknown_border_padding_m")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument("input_topic", default_value="/projected_map"),
        DeclareLaunchArgument("output_topic", default_value="/exploration_grid"),
        DeclareLaunchArgument("inflation_radius_m", default_value="0.1"),
        DeclareLaunchArgument("occupied_threshold", default_value="50"),

        # Add unknown border around /exploration_grid so the frontier detector
        # can treat the finite map edge as unexplored space.
        DeclareLaunchArgument("unknown_border_padding_m", default_value="1.0"),

        DeclareLaunchArgument("use_sim_time", default_value="true"),

        Node(
            package="sim_tb3_velodyne",
            executable="occupancy_grid_inflater",
            name="occupancy_grid_inflater",
            output="screen",
            parameters=[{
                "input_topic": input_topic,
                "output_topic": output_topic,
                "inflation_radius_m": inflation_radius_m,
                "occupied_threshold": occupied_threshold,
                "unknown_border_padding_m": unknown_border_padding_m,
                "use_sim_time": use_sim_time,
            }],
        ),
    ])