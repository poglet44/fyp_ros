from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import UnlessCondition, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context: LaunchContext, *args, **kwargs):
    pkg_share = get_package_share_directory("sim_tb3_velodyne")
    icp_config_file = os.path.join(pkg_share, 'config', 'slam_rtabmap_icp_params.yaml')

    shared_params_file = os.path.join(pkg_share, 'config', 'slam_rtabmap_params.yaml')
    with open(shared_params_file, 'r') as f:
        import yaml
        shared_params = yaml.safe_load(f)

    frame_id = LaunchConfiguration("frame_id")
    use_sim_time = LaunchConfiguration("use_sim_time")

    imu_topic = LaunchConfiguration("imu_topic")
    imu_used = imu_topic.perform(context) != ""

    rgbd_image_topic = LaunchConfiguration("rgbd_image_topic")
    rgbd_images_topic = LaunchConfiguration("rgbd_images_topic")
    rgbd_image_used = (
        rgbd_image_topic.perform(context) != "" or
        rgbd_images_topic.perform(context) != ""
    )
    rgbd_cameras = 0 if rgbd_images_topic.perform(context) != "" else 1

    lidar_topic = LaunchConfiguration("lidar_topic")
    lidar_topic_value = lidar_topic.perform(context)
    lidar_topic_deskewed = lidar_topic_value + "/deskewed"

    localization = LaunchConfiguration("localization").perform(context) in ["true", "True"]
    deskewing = LaunchConfiguration("deskewing").perform(context) in ["true", "True"]
    deskewing_slerp = LaunchConfiguration("deskewing_slerp").perform(context) in ["true", "True"]
    use_external_odom = LaunchConfiguration("use_external_odom").perform(context) in ["true", "True"]
    approx_sync = LaunchConfiguration("approx_sync").perform(context) in ["true", "True"]

    odom_topic = LaunchConfiguration("odom_topic")
    odom_frame_id = LaunchConfiguration("odom_frame_id")
    map_frame_id = LaunchConfiguration("map_frame_id")

    fixed_frame_from_imu = False
    fixed_frame_id = LaunchConfiguration("fixed_frame_id").perform(context)
    if not fixed_frame_id and imu_used:
        fixed_frame_from_imu = True
        fixed_frame_id = frame_id.perform(context) + "_stabilized"

    if not fixed_frame_id or not deskewing:
        lidar_topic_deskewed = lidar_topic

    remappings = []
    if imu_used:
        remappings.append(("imu", imu_topic))
    else:
        remappings.append(("imu", "imu_not_used"))

    if rgbd_image_used:
        if rgbd_cameras == 1:
            remappings.append(("rgbd_image", LaunchConfiguration("rgbd_image_topic")))
        else:
            remappings.append(("rgbd_images", LaunchConfiguration("rgbd_images_topic")))
            
            
            
    cloud_output_voxelized_value = LaunchConfiguration("cloud_output_voxelized").perform(context)
    cloud_subtract_filtering_value = LaunchConfiguration("cloud_subtract_filtering").perform(context)
    cloud_subtract_filtering_min_neighbors_value = LaunchConfiguration("cloud_subtract_filtering_min_neighbors").perform(context)
    scan_cloud_max_points_value = LaunchConfiguration("scan_cloud_max_points").perform(context)
    scan_cloud_is_2d_value = LaunchConfiguration("scan_cloud_is_2d").perform(context)

    map_filter_radius_value = LaunchConfiguration("map_filter_radius").perform(context)
    map_filter_angle_value = LaunchConfiguration("map_filter_angle").perform(context)
    map_cleanup_value = LaunchConfiguration("map_cleanup").perform(context)
    map_always_update_value = LaunchConfiguration("map_always_update").perform(context)
    map_empty_ray_tracing_value = LaunchConfiguration("map_empty_ray_tracing").perform(context)
    octomap_tree_depth_value = LaunchConfiguration("octomap_tree_depth").perform(context)

    common_runtime = {
        "use_sim_time": use_sim_time,
        "frame_id": frame_id,
        "approx_sync": approx_sync,
    }

    icp_runtime = {
        "expected_update_rate": float(LaunchConfiguration("expected_update_rate").perform(context)),
        "deskewing": not fixed_frame_id and deskewing,
        "odom_frame_id": odom_frame_id,
        "guess_frame_id": fixed_frame_id,
        "deskewing_slerp": deskewing_slerp,
    }
    if imu_used:
        icp_runtime["wait_imu_to_init"] = True

    rtabmap_runtime = {
        "map_frame_id": map_frame_id,
        "subscribe_rgbd": rgbd_image_used,
        "rgbd_cameras": rgbd_cameras,
        "cloud_output_voxelized": cloud_output_voxelized_value in ["true", "True"],
        "cloud_subtract_filtering": cloud_subtract_filtering_value in ["true", "True"],
        "cloud_subtract_filtering_min_neighbors": int(cloud_subtract_filtering_min_neighbors_value),
        "scan_cloud_max_points": int(scan_cloud_max_points_value),
        "scan_cloud_is_2d": scan_cloud_is_2d_value in ["true", "True"],

        "map_filter_radius": float(map_filter_radius_value),
        "map_filter_angle": float(map_filter_angle_value),
        "map_cleanup": map_cleanup_value in ["true", "True"],
        "map_always_update": map_always_update_value in ["true", "True"],
        "map_empty_ray_tracing": map_empty_ray_tracing_value in ["true", "True"],
        "octomap_tree_depth": int(octomap_tree_depth_value),
    }

    if use_external_odom:
        rtabmap_runtime["subscribe_odom_info"] = False
        rtabmap_runtime["odom_sensor_sync"] = False
    else:
        rtabmap_runtime["subscribe_odom_info"] = True
        rtabmap_runtime["odom_sensor_sync"] = True

    arguments = []
    if localization:
        rtabmap_runtime["Mem/IncrementalMemory"] = "False"
        rtabmap_runtime["Mem/InitWMWithAllNodes"] = "True"
    else:
        arguments.append("-d")

    nodes = []

    nodes.append(
        Node(
            condition=UnlessCondition(LaunchConfiguration("use_external_odom")),
            package="rtabmap_odom",
            executable="icp_odometry",
            name="icp_odometry",
            output="screen",
            parameters=[icp_config_file, common_runtime, icp_runtime],
            remappings=remappings + [("scan_cloud", lidar_topic_deskewed)],
        )
    )

    if use_external_odom:
        rtabmap_remaps = remappings + [
            ("scan_cloud", lidar_topic_deskewed),
            ("odom", odom_topic),
        ]
        viz_scan_topic = lidar_topic_deskewed
    else:
        rtabmap_remaps = remappings + [
            ("scan_cloud", lidar_topic_deskewed),
            ("odom", "odom"),
        ]
        viz_scan_topic = "odom_filtered_input_scan"

    nodes.append(
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[shared_params, common_runtime, rtabmap_runtime],
            remappings=rtabmap_remaps,
            arguments=arguments,
        )
    )

    nodes.append(
    Node(
        condition=IfCondition(LaunchConfiguration("use_rtabmap_viz")),
        package="rtabmap_viz",
        executable="rtabmap_viz",
        name="rtabmap_viz",
        output="screen",
        parameters=[shared_params, common_runtime, rtabmap_runtime],
        remappings=remappings + [
            ("scan_cloud", viz_scan_topic),
            ("odom", odom_topic if use_external_odom else "odom"),
        ],
    )
)

    if fixed_frame_from_imu:
        nodes.append(
            Node(
                package="rtabmap_util",
                executable="imu_to_tf",
                name="imu_to_tf",
                output="screen",
                parameters=[{
                    "use_sim_time": use_sim_time,
                    "fixed_frame_id": fixed_frame_id,
                    "base_frame_id": frame_id,
                    "wait_for_transform_duration": 0.001,
                }],
                remappings=[("imu/data", imu_topic)],
            )
        )

    if fixed_frame_id and deskewing:
        nodes.append(
            Node(
                package="rtabmap_util",
                executable="lidar_deskewing",
                name="lidar_deskewing",
                output="screen",
                parameters=[{
                    "use_sim_time": use_sim_time,
                    "fixed_frame_id": fixed_frame_id,
                    "wait_for_transform": 0.2,
                    "slerp": deskewing_slerp,
                }],
                remappings=[("input_cloud", lidar_topic)],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_rtabmap_viz", default_value="false"),
        DeclareLaunchArgument("deskewing", default_value="false"),
        DeclareLaunchArgument("deskewing_slerp", default_value="false"),
        DeclareLaunchArgument("frame_id", default_value="base_footprint"),
        DeclareLaunchArgument("fixed_frame_id", default_value=""),
        DeclareLaunchArgument("localization", default_value="false"),
        DeclareLaunchArgument("lidar_topic", default_value="/points"),
        DeclareLaunchArgument("imu_topic", default_value=""),
        DeclareLaunchArgument("rgbd_image_topic", default_value=""),
        DeclareLaunchArgument("rgbd_images_topic", default_value=""),
        DeclareLaunchArgument("expected_update_rate", default_value="15.0"),
        DeclareLaunchArgument("use_external_odom", default_value="true"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("odom_frame_id", default_value="icp_odom"),
        DeclareLaunchArgument("map_frame_id", default_value="map"),
        DeclareLaunchArgument("approx_sync", default_value="true"),
        DeclareLaunchArgument("cloud_output_voxelized", default_value="true"),
        DeclareLaunchArgument("cloud_subtract_filtering", default_value="false"),
        DeclareLaunchArgument("cloud_subtract_filtering_min_neighbors", default_value="5"),
        DeclareLaunchArgument("scan_cloud_max_points", default_value="0"),
        DeclareLaunchArgument("scan_cloud_is_2d", default_value="false"),

        DeclareLaunchArgument("map_filter_radius", default_value="0.0"),
        DeclareLaunchArgument("map_filter_angle", default_value="30.0"),
        DeclareLaunchArgument("map_cleanup", default_value="true"),
        DeclareLaunchArgument("map_always_update", default_value="false"),
        DeclareLaunchArgument("map_empty_ray_tracing", default_value="true"),
        DeclareLaunchArgument("octomap_tree_depth", default_value="16"),
        
        OpaqueFunction(function=launch_setup),
    ])