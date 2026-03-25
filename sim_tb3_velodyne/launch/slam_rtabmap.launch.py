from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    localization = LaunchConfiguration("localization")
    lidar_topic = LaunchConfiguration("lidar_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    frame_id = LaunchConfiguration("frame_id")
    map_frame_id = LaunchConfiguration("map_frame_id")
    qos = LaunchConfiguration("qos")

    parameters = {
        "frame_id": frame_id,
        "map_frame_id": map_frame_id,
        "use_sim_time": use_sim_time,
        "subscribe_rgb": False,
        "subscribe_depth": False,
        "subscribe_rgbd": False,
        "subscribe_scan_cloud": True,
        "subscribe_odom_info": False,
        "odom_sensor_sync": False,
        "qos": qos,
        "wait_for_transform": 0.2,

        # RTAB-Map internal params must be strings
        "Reg/Strategy": "1",
        "Icp/PointToPlane": "true",
        "Icp/VoxelSize": "0.05",
        "Icp/MaxCorrespondenceDistance": "0.5",
        "Icp/CorrespondenceRatio": "0.15",
        "Icp/MaxTranslation": "0.2",
        "Mem/NotLinkedNodesKept": "false",
        "Mem/STMSize": "30",
        "RGBD/AngularUpdate": "0.05",
        "RGBD/LinearUpdate": "0.05",
        "RGBD/CreateOccupancyGrid": "true",
        "RGBD/ProximityMaxGraphDepth": "0",
        "RGBD/ProximityPathMaxNeighbors": "0",
    }

    remappings = [
        ("scan_cloud", lidar_topic),
        ("odom", odom_topic),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("localization", default_value="false"),
        DeclareLaunchArgument("lidar_topic", default_value="/points"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("frame_id", default_value="base_link"),
        DeclareLaunchArgument("map_frame_id", default_value="map"),
        DeclareLaunchArgument("qos", default_value="2"),

        # SLAM mode
        Node(
            condition=UnlessCondition(localization),
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[parameters],
            remappings=remappings,
            arguments=["-d"],
        ),

        # Localization-only mode
        Node(
            condition=IfCondition(localization),
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[parameters, {
                "Mem/IncrementalMemory": "False",
                "Mem/InitWMWithAllNodes": "True",
            }],
            remappings=remappings,
        ),

        Node(
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            parameters=[parameters],
            remappings=remappings,
        ),
    ])