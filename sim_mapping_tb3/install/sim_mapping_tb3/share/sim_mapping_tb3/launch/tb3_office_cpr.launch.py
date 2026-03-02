from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
import os

def generate_launch_description():
    # HARD-CODED WORLD PATH (as you requested)
    WORLD = "/home/thor-usr/Sam/fyp_ws/gazebo_models_worlds_collection/worlds/office_cpr.world"

    # Make Gazebo find your downloaded model folders (adjust if your models live elsewhere)
    MODELS = "/home/thor-usr/Sam/fyp_ws/gazebo_models_worlds_collection/models"

    return LaunchDescription([
        SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=MODELS),

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

