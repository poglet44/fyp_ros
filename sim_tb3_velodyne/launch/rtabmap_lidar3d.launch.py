from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context: LaunchContext, *args, **kwargs):
    frame_id = LaunchConfiguration("frame_id")

    imu_topic = LaunchConfiguration("imu_topic")
    imu_used = imu_topic.perform(context) != ""

    rgbd_image_topic = LaunchConfiguration("rgbd_image_topic")
    rgbd_images_topic = LaunchConfiguration("rgbd_images_topic")
    rgbd_image_used = (
        rgbd_image_topic.perform(context) != "" or
        rgbd_images_topic.perform(context) != ""
    )
    rgbd_cameras = 0 if rgbd_images_topic.perform(context) != "" else 1

    voxel_size = LaunchConfiguration("voxel_size")
    voxel_size_value = float(voxel_size.perform(context))

    use_sim_time = LaunchConfiguration("use_sim_time")

    lidar_topic = LaunchConfiguration("lidar_topic")
    lidar_topic_value = lidar_topic.perform(context)
    lidar_topic_deskewed = lidar_topic_value + "/deskewed"

    localization = LaunchConfiguration("localization").perform(context)
    localization = localization in ["true", "True"]

    deskewing = LaunchConfiguration("deskewing").perform(context)
    deskewing = deskewing in ["true", "True"]

    deskewing_slerp = LaunchConfiguration("deskewing_slerp").perform(context)
    deskewing_slerp = deskewing_slerp in ["true", "True"]

    use_external_odom = LaunchConfiguration("use_external_odom").perform(context)
    use_external_odom = use_external_odom in ["true", "True"]

    odom_topic = LaunchConfiguration("odom_topic")
    odom_frame_id = LaunchConfiguration("odom_frame_id")

    fixed_frame_from_imu = False
    fixed_frame_id = LaunchConfiguration("fixed_frame_id").perform(context)
    if not fixed_frame_id and imu_used:
        fixed_frame_from_imu = True
        fixed_frame_id = frame_id.perform(context) + "_stabilized"

    if not fixed_frame_id or not deskewing:
        lidar_topic_deskewed = lidar_topic

    max_correspondence_distance = voxel_size_value * 10.0

    expected_update_rate_value = LaunchConfiguration("expected_update_rate").perform(context)
    min_loop_closure_overlap_value = LaunchConfiguration("min_loop_closure_overlap").perform(context)

    icp_iterations_value = LaunchConfiguration("icp_iterations").perform(context)
    icp_epsilon_value = LaunchConfiguration("icp_epsilon").perform(context)
    icp_point_to_plane_k_value = LaunchConfiguration("icp_point_to_plane_k").perform(context)
    icp_point_to_plane_radius_value = LaunchConfiguration("icp_point_to_plane_radius").perform(context)
    icp_max_translation_value = LaunchConfiguration("icp_max_translation").perform(context)
    icp_strategy_value = LaunchConfiguration("icp_strategy").perform(context)
    icp_outlier_ratio_value = LaunchConfiguration("icp_outlier_ratio").perform(context)
    icp_correspondence_ratio_value = LaunchConfiguration("icp_correspondence_ratio").perform(context)

    odom_scan_keyframe_thr_value = LaunchConfiguration("odom_scan_keyframe_thr").perform(context)
    odom_scan_max_size_value = LaunchConfiguration("odom_scan_max_size").perform(context)
    odom_bundle_adjustment_value = LaunchConfiguration("odom_bundle_adjustment").perform(context)

    proximity_max_graph_depth_value = LaunchConfiguration("proximity_max_graph_depth").perform(context)
    proximity_path_max_neighbors_value = LaunchConfiguration("proximity_path_max_neighbors").perform(context)
    angular_update_value = LaunchConfiguration("angular_update").perform(context)
    linear_update_value = LaunchConfiguration("linear_update").perform(context)
    create_occupancy_grid_value = LaunchConfiguration("create_occupancy_grid").perform(context)
    not_linked_nodes_kept_value = LaunchConfiguration("not_linked_nodes_kept").perform(context)
    stm_size_value = LaunchConfiguration("stm_size").perform(context)
    reg_strategy_value = LaunchConfiguration("reg_strategy").perform(context)


    shared_parameters = {
        "use_sim_time": use_sim_time,
        "frame_id": frame_id,
        "qos": LaunchConfiguration("qos"),
        "approx_sync": rgbd_image_used,
        "wait_for_transform": 0.2,
        # RTAB-Map internal params are strings
        "Icp/PointToPlane": "true",
        "Icp/Iterations": icp_iterations_value,
        "Icp/VoxelSize": str(voxel_size_value),
        "Icp/Epsilon": icp_epsilon_value,
        "Icp/PointToPlaneK": icp_point_to_plane_k_value,
        "Icp/PointToPlaneRadius": icp_point_to_plane_radius_value,
        "Icp/MaxTranslation": icp_max_translation_value,
        "Icp/MaxCorrespondenceDistance": str(max_correspondence_distance),
        "Icp/Strategy": icp_strategy_value,
        "Icp/OutlierRatio": icp_outlier_ratio_value,
    }

    icp_odometry_parameters = {
        "expected_update_rate": float(expected_update_rate_value),
        "deskewing": not fixed_frame_id and deskewing,
        "odom_frame_id": odom_frame_id,
        "guess_frame_id": fixed_frame_id,
        "deskewing_slerp": deskewing_slerp,
        # RTAB-Map internal params are strings
        "Odom/ScanKeyFrameThr": odom_scan_keyframe_thr_value,
        "OdomF2M/ScanSubtractRadius": str(voxel_size_value),
        "OdomF2M/ScanMaxSize": odom_scan_max_size_value,
        "OdomF2M/BundleAdjustment": odom_bundle_adjustment_value,
        "Icp/CorrespondenceRatio": icp_correspondence_ratio_value,
    }


    if imu_used:
        icp_odometry_parameters["wait_imu_to_init"] = True

    # SLAM parameters differ slightly depending on odom source
    rtabmap_parameters = {
        "subscribe_depth": False,
        "subscribe_rgb": False,
        "subscribe_scan_cloud": True,
        "map_frame_id": LaunchConfiguration("map_frame_id"),
        # RTAB-Map internal params are strings
        "RGBD/ProximityMaxGraphDepth": proximity_max_graph_depth_value,
        "RGBD/ProximityPathMaxNeighbors": proximity_path_max_neighbors_value,
        "RGBD/AngularUpdate": angular_update_value,
        "RGBD/LinearUpdate": linear_update_value,
        "RGBD/CreateOccupancyGrid": create_occupancy_grid_value,
        "Mem/NotLinkedNodesKept": not_linked_nodes_kept_value,
        "Mem/STMSize": stm_size_value,
        "Reg/Strategy": reg_strategy_value,
        "Icp/CorrespondenceRatio": min_loop_closure_overlap_value,
    }
    if use_external_odom:
        rtabmap_parameters["subscribe_odom_info"] = False
        rtabmap_parameters["odom_sensor_sync"] = False
    else:
        rtabmap_parameters["subscribe_odom_info"] = True
        rtabmap_parameters["odom_sensor_sync"] = True

    arguments = []
    if localization:
        rtabmap_parameters["Mem/IncrementalMemory"] = "False"
        rtabmap_parameters["Mem/InitWMWithAllNodes"] = "True"
    else:
        arguments.append("-d")

    # Common remaps
    remappings = []
    if imu_used:
        remappings.append(("imu", LaunchConfiguration("imu_topic")))
    else:
        remappings.append(("imu", "imu_not_used"))

    if rgbd_image_used:
        if rgbd_cameras == 1:
            remappings.append(("rgbd_image", LaunchConfiguration("rgbd_image_topic")))
        else:
            remappings.append(("rgbd_images", LaunchConfiguration("rgbd_images_topic")))

    nodes = []

    # ICP odometry path
    nodes.append(
        Node(
            condition=UnlessCondition(LaunchConfiguration("use_external_odom")),
            package="rtabmap_odom",
            executable="icp_odometry",
            output="screen",
            parameters=[shared_parameters, icp_odometry_parameters],
            remappings=remappings + [
                ("scan_cloud", lidar_topic_deskewed),
            ],
        )
    )

    # RTAB-Map SLAM
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
            output="screen",
            parameters=[
                shared_parameters,
                rtabmap_parameters,
                {
                    "subscribe_rgbd": rgbd_image_used,
                    "rgbd_cameras": rgbd_cameras,
                },
            ],
            remappings=rtabmap_remaps,
            arguments=arguments,
        )
    )

    nodes.append(
        Node(
            package="rtabmap_viz",
            executable="rtabmap_viz",
            output="screen",
            parameters=[shared_parameters, rtabmap_parameters],
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
                output="screen",
                parameters=[{
                    "use_sim_time": use_sim_time,
                    "fixed_frame_id": fixed_frame_id,
                    "wait_for_transform": 0.2,
                    "slerp": deskewing_slerp,
                }],
                remappings=[
                    ("input_cloud", lidar_topic),
                ],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulated clock.",
        ),
        DeclareLaunchArgument(
            "deskewing",
            default_value="false",
            description="Enable lidar deskewing.",
        ),
        DeclareLaunchArgument(
            "deskewing_slerp",
            default_value="false",
            description="Use SLERP during deskewing.",
        ),
        DeclareLaunchArgument(
            "frame_id",
            default_value="base_footprint",
            description="Base frame of the robot.",
        ),
        DeclareLaunchArgument(
            "fixed_frame_id",
            default_value="",
            description="Fixed frame used for lidar deskewing. If empty and IMU is used, one will be generated.",
        ),
        DeclareLaunchArgument(
            "localization",
            default_value="false",
            description="Localization mode.",
        ),
        DeclareLaunchArgument(
            "lidar_topic",
            default_value="/points",
            description="PointCloud2 lidar topic.",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="",
            description="Optional IMU topic.",
        ),
        DeclareLaunchArgument(
            "rgbd_image_topic",
            default_value="",
            description="Optional RGBD image topic.",
        ),
        DeclareLaunchArgument(
            "rgbd_images_topic",
            default_value="",
            description="Optional multi-RGBD images topic.",
        ),
        DeclareLaunchArgument(
            "expected_update_rate",
            default_value="15.0",
            description="Expected lidar frame rate. Slightly above actual is typical.",
        ),
        DeclareLaunchArgument(
            "voxel_size",
            default_value="0.05",
            description="Voxel size in meters.",
        ),
        DeclareLaunchArgument(
            "min_loop_closure_overlap",
            default_value="0.15",
            description="Minimum scan overlap ratio for loop closure.",
        ),
        DeclareLaunchArgument(
            "qos",
            default_value="2",
            description="QoS setting passed to RTAB-Map.",
        ),

        # New odometry control
        DeclareLaunchArgument(
            "use_external_odom",
            default_value="false",
            description="Use external odometry topic instead of RTAB-Map ICP odometry.",
        ),
        DeclareLaunchArgument(
            "odom_topic",
            default_value="/odom",
            description="External odometry topic to use when use_external_odom:=true.",
        ),
        DeclareLaunchArgument(
            "odom_frame_id",
            default_value="icp_odom",
            description="Odometry frame id used by icp_odometry when use_external_odom:=false.",
        ),
        DeclareLaunchArgument(
            "map_frame_id",
            default_value="new_map",
            description="Global map frame published by RTAB-Map.",
        ),

        # Exposed RTAB-Map / ICP params
        DeclareLaunchArgument("icp_iterations", default_value="10"),
        DeclareLaunchArgument("icp_epsilon", default_value="0.001"),
        DeclareLaunchArgument("icp_point_to_plane_k", default_value="20"),
        DeclareLaunchArgument("icp_point_to_plane_radius", default_value="0"),
        DeclareLaunchArgument("icp_max_translation", default_value="3"),
        DeclareLaunchArgument("icp_strategy", default_value="1"),
        DeclareLaunchArgument("icp_outlier_ratio", default_value="0.7"),
        DeclareLaunchArgument("icp_correspondence_ratio", default_value="0.01"),
        DeclareLaunchArgument("odom_scan_keyframe_thr", default_value="0.4"),
        DeclareLaunchArgument("odom_scan_max_size", default_value="15000"),
        DeclareLaunchArgument("odom_bundle_adjustment", default_value="false"),
        DeclareLaunchArgument("proximity_max_graph_depth", default_value="0"),
        DeclareLaunchArgument("proximity_path_max_neighbors", default_value="1"),
        DeclareLaunchArgument("angular_update", default_value="0.05"),
        DeclareLaunchArgument("linear_update", default_value="0.05"),
        DeclareLaunchArgument("create_occupancy_grid", default_value="false"),
        DeclareLaunchArgument("not_linked_nodes_kept", default_value="false"),
        DeclareLaunchArgument("stm_size", default_value="30"),
        DeclareLaunchArgument("reg_strategy", default_value="1"),

        OpaqueFunction(function=launch_setup),
    ])