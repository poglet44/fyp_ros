from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    icp_max_rotation_value = LaunchConfiguration("icp_max_rotation").perform(context)
    icp_strategy_value = LaunchConfiguration("icp_strategy").perform(context)
    icp_outlier_ratio_value = LaunchConfiguration("icp_outlier_ratio").perform(context)
    icp_correspondence_ratio_value = LaunchConfiguration("icp_correspondence_ratio").perform(context)

    odom_scan_keyframe_thr_value = LaunchConfiguration("odom_scan_keyframe_thr").perform(context)
    odom_scan_max_size_value = LaunchConfiguration("odom_scan_max_size").perform(context)
    odom_bundle_adjustment_value = LaunchConfiguration("odom_bundle_adjustment").perform(context)

    proximity_max_graph_depth_value = LaunchConfiguration("proximity_max_graph_depth").perform(context)
    proximity_path_max_neighbors_value = LaunchConfiguration("proximity_path_max_neighbors").perform(context)
    proximity_by_space_value = LaunchConfiguration("proximity_by_space").perform(context)
    neighbor_link_refining_value = LaunchConfiguration("neighbor_link_refining").perform(context)
    optimize_from_graph_end_value = LaunchConfiguration("optimize_from_graph_end").perform(context)

    angular_update_value = LaunchConfiguration("angular_update").perform(context)
    linear_update_value = LaunchConfiguration("linear_update").perform(context)
    create_occupancy_grid_value = LaunchConfiguration("create_occupancy_grid").perform(context)
    not_linked_nodes_kept_value = LaunchConfiguration("not_linked_nodes_kept").perform(context)
    stm_size_value = LaunchConfiguration("stm_size").perform(context)
    reg_strategy_value = LaunchConfiguration("reg_strategy").perform(context)
    reg_force_3dof_value = LaunchConfiguration("reg_force_3dof").perform(context)

    grid_sensor_value = LaunchConfiguration("grid_sensor").perform(context)
    grid_range_min_value = LaunchConfiguration("grid_range_min").perform(context)
    grid_range_max_value = LaunchConfiguration("grid_range_max").perform(context)
    grid_ray_tracing_value = LaunchConfiguration("grid_ray_tracing").perform(context)

    grid_cell_size_value = LaunchConfiguration("grid_cell_size").perform(context)
    grid_footprint_length_value = LaunchConfiguration("grid_footprint_length").perform(context)
    grid_footprint_width_value = LaunchConfiguration("grid_footprint_width").perform(context)
    grid_footprint_height_value = LaunchConfiguration("grid_footprint_height").perform(context)
    grid_normals_segmentation_value = LaunchConfiguration("grid_normals_segmentation").perform(context)
    grid_max_obstacle_height_value = LaunchConfiguration("grid_max_obstacle_height").perform(context)
    grid_min_ground_height_value = LaunchConfiguration("grid_min_ground_height").perform(context)
    grid_max_ground_height_value = LaunchConfiguration("grid_max_ground_height").perform(context)
    grid_max_ground_angle_value = LaunchConfiguration("grid_max_ground_angle").perform(context)
    grid_normal_k_value = LaunchConfiguration("grid_normal_k").perform(context)
    grid_scan_decimation_value = LaunchConfiguration("grid_scan_decimation").perform(context)
    grid_pre_voxel_filtering_value = LaunchConfiguration("grid_pre_voxel_filtering").perform(context)
    grid_map_frame_projection_value = LaunchConfiguration("grid_map_frame_projection").perform(context)
    grid_flat_obstacle_detected_value = LaunchConfiguration("grid_flat_obstacle_detected").perform(context)
    grid_cluster_radius_value = LaunchConfiguration("grid_cluster_radius").perform(context)
    grid_min_cluster_size_value = LaunchConfiguration("grid_min_cluster_size").perform(context)
    detection_rate_value = LaunchConfiguration("detection_rate").perform(context)
    approx_sync_value = LaunchConfiguration("approx_sync").perform(context)
    approx_sync_value = approx_sync_value in ["true", "True"]

    shared_parameters = {
        "use_sim_time": use_sim_time,
        "frame_id": frame_id,
        "qos": LaunchConfiguration("qos"),
        "approx_sync": approx_sync_value,
        "wait_for_transform": 0.2,
        "Icp/PointToPlane": ParameterValue("true", value_type=str),
        "Icp/Iterations": ParameterValue(icp_iterations_value, value_type=str),
        "Icp/VoxelSize": ParameterValue(str(voxel_size_value), value_type=str),
        "Icp/Epsilon": ParameterValue(icp_epsilon_value, value_type=str),
        "Icp/PointToPlaneK": ParameterValue(icp_point_to_plane_k_value, value_type=str),
        "Icp/PointToPlaneRadius": ParameterValue(icp_point_to_plane_radius_value, value_type=str),
        "Icp/MaxTranslation": ParameterValue(icp_max_translation_value, value_type=str),
        "Icp/MaxRotation": ParameterValue(icp_max_rotation_value, value_type=str),
        "Icp/MaxCorrespondenceDistance": ParameterValue(str(max_correspondence_distance), value_type=str),
        "Icp/Strategy": ParameterValue(icp_strategy_value, value_type=str),
        "Icp/OutlierRatio": ParameterValue(icp_outlier_ratio_value, value_type=str),
        "Reg/Force3DoF": ParameterValue(reg_force_3dof_value, value_type=str),
        "Grid/Sensor": ParameterValue(grid_sensor_value, value_type=str),
        "Grid/RangeMin": ParameterValue(grid_range_min_value, value_type=str),
        "Grid/RangeMax": ParameterValue(grid_range_max_value, value_type=str),
        "Grid/RayTracing": ParameterValue(grid_ray_tracing_value, value_type=str),
        "Grid/CellSize": ParameterValue(grid_cell_size_value, value_type=str),
        "Grid/FootprintLength": ParameterValue(grid_footprint_length_value, value_type=str),
        "Grid/FootprintWidth": ParameterValue(grid_footprint_width_value, value_type=str),
        "Grid/FootprintHeight": ParameterValue(grid_footprint_height_value, value_type=str),
        "Grid/NormalsSegmentation": ParameterValue(grid_normals_segmentation_value, value_type=str),
        "Grid/MaxObstacleHeight": ParameterValue(grid_max_obstacle_height_value, value_type=str),
        "Grid/MinGroundHeight": ParameterValue(grid_min_ground_height_value, value_type=str),
        "Grid/MaxGroundHeight": ParameterValue(grid_max_ground_height_value, value_type=str),
        "Grid/MaxGroundAngle": ParameterValue(grid_max_ground_angle_value, value_type=str),
        "Grid/NormalK": ParameterValue(grid_normal_k_value, value_type=str),
        "Grid/ScanDecimation": ParameterValue(grid_scan_decimation_value, value_type=str),
        "Grid/PreVoxelFiltering": ParameterValue(grid_pre_voxel_filtering_value, value_type=str),
        "Grid/MapFrameProjection": ParameterValue(grid_map_frame_projection_value, value_type=str),
        "Grid/FlatObstacleDetected": ParameterValue(grid_flat_obstacle_detected_value, value_type=str),
        "Grid/ClusterRadius": ParameterValue(grid_cluster_radius_value, value_type=str),
        "Grid/MinClusterSize": ParameterValue(grid_min_cluster_size_value, value_type=str),
    }

    icp_odometry_parameters = {
        "expected_update_rate": float(expected_update_rate_value),
        "deskewing": not fixed_frame_id and deskewing,
        "odom_frame_id": odom_frame_id,
        "guess_frame_id": fixed_frame_id,
        "deskewing_slerp": deskewing_slerp,
        "Odom/ScanKeyFrameThr": ParameterValue(odom_scan_keyframe_thr_value, value_type=str),
        "OdomF2M/ScanSubtractRadius": ParameterValue(str(voxel_size_value), value_type=str),
        "OdomF2M/ScanMaxSize": ParameterValue(odom_scan_max_size_value, value_type=str),
        "OdomF2M/BundleAdjustment": ParameterValue(odom_bundle_adjustment_value, value_type=str),
        "Icp/CorrespondenceRatio": ParameterValue(icp_correspondence_ratio_value, value_type=str),
    }

    if imu_used:
        icp_odometry_parameters["wait_imu_to_init"] = True

    rtabmap_parameters = {
        "subscribe_depth": False,
        "subscribe_rgb": False,
        "subscribe_scan_cloud": True,
        "map_frame_id": LaunchConfiguration("map_frame_id"),
        "RGBD/ProximityMaxGraphDepth": ParameterValue(proximity_max_graph_depth_value, value_type=str),
        "RGBD/ProximityPathMaxNeighbors": ParameterValue(proximity_path_max_neighbors_value, value_type=str),
        "RGBD/ProximityBySpace": ParameterValue(proximity_by_space_value, value_type=str),
        "RGBD/NeighborLinkRefining": ParameterValue(neighbor_link_refining_value, value_type=str),
        "RGBD/OptimizeFromGraphEnd": ParameterValue(optimize_from_graph_end_value, value_type=str),
        "RGBD/AngularUpdate": ParameterValue(angular_update_value, value_type=str),
        "RGBD/LinearUpdate": ParameterValue(linear_update_value, value_type=str),
        "RGBD/CreateOccupancyGrid": ParameterValue(create_occupancy_grid_value, value_type=str),
        "Mem/NotLinkedNodesKept": ParameterValue(not_linked_nodes_kept_value, value_type=str),
        "Mem/STMSize": ParameterValue(stm_size_value, value_type=str),
        "Reg/Strategy": ParameterValue(reg_strategy_value, value_type=str),
        "Icp/CorrespondenceRatio": ParameterValue(min_loop_closure_overlap_value, value_type=str),
        "Rtabmap/DetectionRate": ParameterValue(detection_rate_value, value_type=str),
    }
    if use_external_odom:
        rtabmap_parameters["subscribe_odom_info"] = False
        rtabmap_parameters["odom_sensor_sync"] = False
    else:
        rtabmap_parameters["subscribe_odom_info"] = True
        rtabmap_parameters["odom_sensor_sync"] = True

    arguments = []
    if localization:
        rtabmap_parameters["Mem/IncrementalMemory"] = ParameterValue("False", value_type=str)
        rtabmap_parameters["Mem/InitWMWithAllNodes"] = ParameterValue("True", value_type=str)
    else:
        arguments.append("-d")

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
        DeclareLaunchArgument(
            "use_external_odom",
            default_value="true",
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
            default_value="map",
            description="Global map frame published by RTAB-Map.",
        ),
        DeclareLaunchArgument("icp_iterations", default_value="10"),
        DeclareLaunchArgument("icp_epsilon", default_value="0.001"),
        DeclareLaunchArgument("icp_point_to_plane_k", default_value="20"),
        DeclareLaunchArgument("icp_point_to_plane_radius", default_value="0"),
        DeclareLaunchArgument("icp_max_translation", default_value="0.2"),
        DeclareLaunchArgument("icp_max_rotation", default_value="0.78"),
        DeclareLaunchArgument("icp_strategy", default_value="1"),
        DeclareLaunchArgument("icp_outlier_ratio", default_value="0.85"),
        DeclareLaunchArgument("icp_correspondence_ratio", default_value="0.01"),
        DeclareLaunchArgument("odom_scan_keyframe_thr", default_value="0.4"),
        DeclareLaunchArgument("odom_scan_max_size", default_value="15000"),
        DeclareLaunchArgument("odom_bundle_adjustment", default_value="false"),
        DeclareLaunchArgument("proximity_max_graph_depth", default_value="0"),
        DeclareLaunchArgument("proximity_path_max_neighbors", default_value="0"),
        DeclareLaunchArgument("proximity_by_space", default_value="false"),
        DeclareLaunchArgument("neighbor_link_refining", default_value="true"),
        DeclareLaunchArgument("optimize_from_graph_end", default_value="false"),
        DeclareLaunchArgument("angular_update", default_value="0.02"),
        DeclareLaunchArgument("linear_update", default_value="0.02"),
        DeclareLaunchArgument("create_occupancy_grid", default_value="true"),
        DeclareLaunchArgument("not_linked_nodes_kept", default_value="false"),
        DeclareLaunchArgument("stm_size", default_value="30"),
        DeclareLaunchArgument("reg_strategy", default_value="1"),
        DeclareLaunchArgument("reg_force_3dof", default_value="true"),
        DeclareLaunchArgument("grid_sensor", default_value="0"),
        DeclareLaunchArgument("grid_range_min", default_value="0.3"),
        DeclareLaunchArgument("grid_range_max", default_value="0"),
        DeclareLaunchArgument("grid_ray_tracing", default_value="true"),
        DeclareLaunchArgument("grid_cell_size", default_value="0.05"),
        DeclareLaunchArgument("grid_footprint_length", default_value="0.18"),
        DeclareLaunchArgument("grid_footprint_width", default_value="0.18"),
        DeclareLaunchArgument("grid_footprint_height", default_value="0.3"),
        DeclareLaunchArgument("grid_normals_segmentation", default_value="true"),
        DeclareLaunchArgument("grid_max_obstacle_height", default_value="1.0"),
        DeclareLaunchArgument("grid_min_ground_height", default_value="-0.1"),
        DeclareLaunchArgument("grid_max_ground_height", default_value="0.1"),
        DeclareLaunchArgument("grid_max_ground_angle", default_value="45"),
        DeclareLaunchArgument("grid_normal_k", default_value="20"),
        DeclareLaunchArgument("grid_scan_decimation", default_value="1"),
        DeclareLaunchArgument("grid_pre_voxel_filtering", default_value="true"),
        DeclareLaunchArgument("grid_map_frame_projection", default_value="false"),
        DeclareLaunchArgument("grid_flat_obstacle_detected", default_value="true"),
        DeclareLaunchArgument("grid_cluster_radius", default_value="0.15"),
        DeclareLaunchArgument("grid_min_cluster_size", default_value="5"),
        DeclareLaunchArgument("detection_rate", default_value="15.0"),
        DeclareLaunchArgument("approx_sync", default_value="true"),
        OpaqueFunction(function=launch_setup),
    ])