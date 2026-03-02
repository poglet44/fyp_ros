from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition


def generate_launch_description():
    # ---- HARD-CODED PATHS (edit these if you move folders) ----
    WORLD = "/home/thor-usr/Sam/fyp_ws/gazebo_models_worlds_collection/worlds/office_cpr.world"

    # TurtleBot3 model (we'll spawn the burger SDF directly)
    TB3_MODEL_SDF = "/home/thor-usr/Sam/fyp_ws/sim_models/turtlebot3_burger/model.sdf"


    # Gazebo launch files
    GZSERVER_LAUNCH = "/opt/ros/humble/share/gazebo_ros/launch/gzserver.launch.py"
    GZCLIENT_LAUNCH = "/opt/ros/humble/share/gazebo_ros/launch/gzclient.launch.py"

    # TurtleBot3 robot_state_publisher launch (provides TF / robot_description)
    TB3_RSP_LAUNCH = "/opt/ros/humble/share/turtlebot3_gazebo/launch/robot_state_publisher.launch.py"

    # Your models folder + TB3 models folder
    MODELS = (
        "/opt/ros/humble/share/turtlebot3_gazebo/models:"
        "/home/thor-usr/Sam/fyp_ws/gazebo_models_worlds_collection/models"
    )

    # ---- SIM TIME ----
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    
    # ---- USE GUI ----
    gui = LaunchConfiguration("gui", default="true")
    
    declare_gui = DeclareLaunchArgument(
    "gui",
    default_value="true",
    description="Whether to start Gazebo GUI (gzclient)."
)

    # ---- SET ENV ----
    set_model_path = SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=MODELS)

    # Some setups try to hit online model DB; forcing empty avoids surprises
    disable_model_db = SetEnvironmentVariable(name="GAZEBO_MODEL_DATABASE_URI", value="")

    # TurtleBot3 tooling uses this env var in places; keep it consistent
    set_tb3_model = SetEnvironmentVariable(name="TURTLEBOT3_MODEL", value="burger")

    # ---- START GAZEBO (server + client) ----
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(GZSERVER_LAUNCH),
        launch_arguments={"world": WORLD}.items(),
    )

    gzclient = ExecuteProcess(
    condition=IfCondition(gui),
    cmd=["gzclient"],
    output="screen"
    )

    # ---- ROBOT STATE PUBLISHER (TF) ----
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(TB3_RSP_LAUNCH),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # ---- SPAWN TURTLEBOT3 INTO GAZEBO ----
    spawn_tb3 = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity", "tb3",
            "-file", TB3_MODEL_SDF,
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.01",
            "-Y", "0.0",
        ],
    )

    # Give Gazebo a moment to bring up /spawn_entity service
    spawn_after_gazebo = TimerAction(period=3.0, actions=[spawn_tb3])

    return LaunchDescription([
        declare_gui,
        set_model_path,
        disable_model_db,
        set_tb3_model,
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_after_gazebo,
    ])

