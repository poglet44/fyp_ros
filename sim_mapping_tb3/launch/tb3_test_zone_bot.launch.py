from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node

def generate_launch_description():

    WORLD = "/home/thor-usr/Sam/fyp_ws/gazebo_models_worlds_collection/worlds/test_zone.world"
    MODELS = (
    "/opt/ros/humble/share/turtlebot3_gazebo/models:"
    "/home/thor-usr/Sam/fyp_ws/gazebo_models_worlds_collection/models"
)


    gazebo = ExecuteProcess(
        cmd=[
            "gazebo",
            "--verbose",
            WORLD,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
    )

    spawn_tb3 = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity", "tb3",
            "-database", "turtlebot3_burger",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.01",
            "-Y", "0.0",
        ],
    )

    return LaunchDescription([
        SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=MODELS),
        SetEnvironmentVariable(name="GAZEBO_MODEL_DATABASE_URI", value=""),
        gazebo,
        TimerAction(period=3.0, actions=[spawn_tb3]),
    ])

