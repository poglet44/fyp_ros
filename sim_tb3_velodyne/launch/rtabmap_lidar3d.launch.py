from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    lidar_topic = LaunchConfiguration("lidar_topic")
    frame_id = LaunchConfiguration("frame_id")
    use_sim_time = LaunchConfiguration("use_sim_time")
    deskewing = LaunchConfiguration("deskewing")
    imu_topic = LaunchConfiguration("imu_topic")
    expected_update_rate = LaunchConfiguration("expected_update_rate")
    voxel_size = LaunchConfiguration("voxel_size")
    min_loop_closure_overlap = LaunchConfiguration("min_loop_closure_overlap")
    localization = LaunchConfiguration("localization")
    fixed_frame_id = LaunchConfiguration("fixed_frame_id")
    qos = LaunchConfiguration("qos")

    rtabmap_examples_launch = PathJoinSubstitution([
        FindPackageShare("rtabmap_examples"),
        "launch",
        "lidar3d.launch.py"
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "lidar_topic",
            default_value="/points",
            description="Input PointCloud2 topic for RTAB-Map",
        ),
        DeclareLaunchArgument(
            "frame_id",
            default_value="base_footprint",
            description="Base frame of the lidar / robot for RTAB-Map",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use Gazebo simulated clock",
        ),
        DeclareLaunchArgument(
            "deskewing",
            default_value="false",
            description="Enable lidar deskewing",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="",
            description="Optional IMU topic. Leave empty for lidar-only",
        ),
        DeclareLaunchArgument(
            "expected_update_rate",
            default_value="12.0",
            description="Expected lidar update rate, slightly above actual sensor rate",
        ),
        DeclareLaunchArgument(
            "voxel_size",
            default_value="0.1",
            description="Voxel size for ICP / scan downsampling",
        ),
        DeclareLaunchArgument(
            "min_loop_closure_overlap",
            default_value="0.2",
            description="Minimum overlap ratio for loop closures",
        ),
        DeclareLaunchArgument(
            "localization",
            default_value="false",
            description="Run RTAB-Map in localization mode",
        ),
        DeclareLaunchArgument(
            "fixed_frame_id",
            default_value="",
            description="Optional fixed frame for deskewing",
        ),
        DeclareLaunchArgument(
            "qos",
            default_value="2",
            description="QoS setting passed to RTAB-Map launch",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rtabmap_examples_launch),
            launch_arguments={
                "lidar_topic": lidar_topic,
                "frame_id": frame_id,
                "use_sim_time": use_sim_time,
                "deskewing": deskewing,
                "imu_topic": imu_topic,
                "expected_update_rate": expected_update_rate,
                "voxel_size": voxel_size,
                "min_loop_closure_overlap": min_loop_closure_overlap,
                "localization": localization,
                "fixed_frame_id": fixed_frame_id,
                "qos": qos,
            }.items(),
        ),
    ])