from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    world = LaunchConfiguration("world")

    turtlebot3_gazebo_share = get_package_share_directory("turtlebot3_gazebo")
    gazebo_launch = os.path.join(turtlebot3_gazebo_share, "launch", "turtlebot3_world.launch.py")

    # This is the standard installed path for TB3 house world
    default_house_world = os.path.join(turtlebot3_gazebo_share, "worlds", "turtlebot3_house.world")

    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value=default_house_world,
            description="Full path to world file"
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={"world": world}.items()
        ),
    ])
