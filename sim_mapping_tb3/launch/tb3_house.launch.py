from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    WORLD = "/opt/ros/humble/share/turtlebot3_gazebo/worlds/turtlebot3_house.world"

    tb3_share = get_package_share_directory("turtlebot3_gazebo")
    tb3_models = os.path.join(tb3_share, "models")

    # Your custom models (keep if you want; remove if not needed)
    custom_models = "/home/thor-usr/Sam/fyp_ws/gazebo_models_worlds_collection/models"

    # IMPORTANT: include TB3 models so model:// resolves locally
    model_path = tb3_models + ":" + custom_models

    return LaunchDescription([
        SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=model_path),

        ExecuteProcess(
            cmd=[
                "gazebo",
                "--verbose",
                WORLD,
                "-s", "libgazebo_ros_init.so",
                "-s", "libgazebo_ros_factory.so",
            ],
            output="screen",
        ),
    ])

