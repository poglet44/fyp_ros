from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable

def generate_launch_description():
    # HARD-CODED WORLD PATH
    WORLD = "/home/thor-usr/Sam/fyp_ws/gazebo_models_worlds_collection/worlds/test_zone.world"

    # HARD-CODED MODELS PATH
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

