#!/usr/bin/env python3

import csv
import heapq
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path as FilePath
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


Cell = Tuple[int, int]


@dataclass
class FrontierCandidate:
    cluster_id: int
    goal_cell: Cell
    goal_pose: Pose
    path_cells: List[Cell]
    path_length_m: float
    cluster_size_cells: int
    unknown_gain_cells: int
    score: float = float("inf")


class FrontierDetector(Node):
    def __init__(self):
        super().__init__("frontier_detector")

        # Topics
        self.declare_parameter("map_topic", "/exploration_grid")
        self.declare_parameter("frontier_cells_topic", "/frontier_cells")
        self.declare_parameter("frontier_markers_topic", "/frontier_markers")
        self.declare_parameter("frontier_goals_topic", "/frontier_goals")
        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("frontier_path_topic", "/frontier_path")

        # Frames
        self.declare_parameter("robot_frame", "body")
        self.declare_parameter("tf_lookup_timeout_s", 0.20)
        self.declare_parameter("robot_pose_warn_period_s", 5.0)

        # Occupancy classification
        self.declare_parameter("unknown_value", -1)
        self.declare_parameter("free_max_value", 0)
        self.declare_parameter("occupied_min_value", 50)

        # Frontier filtering
        self.declare_parameter("use_8_connected_frontiers", True)
        self.declare_parameter("min_cluster_size_cells", 8)
        self.declare_parameter("min_obstacle_clearance_m", 0.20)
        self.declare_parameter("min_robot_distance_m", 0.75)

        # Goal generation
        self.declare_parameter("goal_search_radius_m", 1.20)
        self.declare_parameter("max_goals_to_publish", 10)

        # Path planning
        self.declare_parameter("enable_path_planning", True)
        self.declare_parameter("allow_diagonal_motion", True)
        self.declare_parameter("prevent_diagonal_corner_cutting", True)
        self.declare_parameter("path_obstacle_clearance_m", 0.20)

        # Planning throttling
        self.declare_parameter("plan_every_n_maps", 3)
        self.declare_parameter("min_plan_period_s", 0.75)

        # Reachability
        self.declare_parameter("require_reachable", True)

        # Selection policy
        self.declare_parameter("selection_policy", "nearest")
        self.declare_parameter("utility_distance_weight", 1.0)
        self.declare_parameter("utility_gain_weight", 0.02)

        # Goal hysteresis
        self.declare_parameter("enable_goal_hysteresis", True)
        self.declare_parameter("hysteresis_goal_match_distance_m", 0.75)
        self.declare_parameter("hysteresis_switch_margin", 0.75)

        # Visualisation / logging
        self.declare_parameter("marker_scale_m", 0.08)
        self.declare_parameter("publish_debug_every_n_maps", 10)
        self.declare_parameter("publish_text_labels", False)

        # CSV logging
        self.declare_parameter("enable_logging", True)
        self.declare_parameter("log_root_dir", "~/Sam/fyp_ws/logs")

        self.map_topic = self.get_parameter("map_topic").value
        self.frontier_cells_topic = self.get_parameter("frontier_cells_topic").value
        self.frontier_markers_topic = self.get_parameter("frontier_markers_topic").value
        self.frontier_goals_topic = self.get_parameter("frontier_goals_topic").value
        self.selected_goal_topic = self.get_parameter("selected_goal_topic").value
        self.frontier_path_topic = self.get_parameter("frontier_path_topic").value

        self.robot_frame = self.get_parameter("robot_frame").value
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)
        self.robot_pose_warn_period_s = float(self.get_parameter("robot_pose_warn_period_s").value)

        self.unknown_value = int(self.get_parameter("unknown_value").value)
        self.free_max_value = int(self.get_parameter("free_max_value").value)
        self.occupied_min_value = int(self.get_parameter("occupied_min_value").value)

        self.use_8_connected_frontiers = bool(self.get_parameter("use_8_connected_frontiers").value)
        self.min_cluster_size_cells = int(self.get_parameter("min_cluster_size_cells").value)
        self.min_obstacle_clearance_m = float(self.get_parameter("min_obstacle_clearance_m").value)
        self.min_robot_distance_m = float(self.get_parameter("min_robot_distance_m").value)

        self.goal_search_radius_m = float(self.get_parameter("goal_search_radius_m").value)
        self.max_goals_to_publish = int(self.get_parameter("max_goals_to_publish").value)

        self.enable_path_planning = bool(self.get_parameter("enable_path_planning").value)
        self.allow_diagonal_motion = bool(self.get_parameter("allow_diagonal_motion").value)
        self.prevent_diagonal_corner_cutting = bool(
            self.get_parameter("prevent_diagonal_corner_cutting").value
        )
        self.path_obstacle_clearance_m = float(self.get_parameter("path_obstacle_clearance_m").value)

        self.plan_every_n_maps = int(self.get_parameter("plan_every_n_maps").value)
        self.min_plan_period_s = float(self.get_parameter("min_plan_period_s").value)

        self.require_reachable = bool(self.get_parameter("require_reachable").value)

        self.selection_policy = str(self.get_parameter("selection_policy").value)
        self.utility_distance_weight = float(self.get_parameter("utility_distance_weight").value)
        self.utility_gain_weight = float(self.get_parameter("utility_gain_weight").value)

        self.enable_goal_hysteresis = bool(self.get_parameter("enable_goal_hysteresis").value)
        self.hysteresis_goal_match_distance_m = float(
            self.get_parameter("hysteresis_goal_match_distance_m").value
        )
        self.hysteresis_switch_margin = float(self.get_parameter("hysteresis_switch_margin").value)

        self.marker_scale_m = float(self.get_parameter("marker_scale_m").value)
        self.publish_debug_every_n_maps = int(self.get_parameter("publish_debug_every_n_maps").value)
        self.publish_text_labels = bool(self.get_parameter("publish_text_labels").value)

        self.enable_logging = bool(self.get_parameter("enable_logging").value)
        self.log_root_dir = str(self.get_parameter("log_root_dir").value)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.frontier_grid_pub = self.create_publisher(OccupancyGrid, self.frontier_cells_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.frontier_markers_topic, 10)
        self.goal_pub = self.create_publisher(PoseArray, self.frontier_goals_topic, 10)
        self.selected_goal_pub = self.create_publisher(PoseStamped, self.selected_goal_topic, 10)
        self.path_pub = self.create_publisher(Path, self.frontier_path_topic, 10)

        # Queue depth 1 is deliberate: frontier detection should use the newest map,
        # not process stale queued maps.
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 1)

        self.map_count = 0
        self.previous_selected_goal_cell: Optional[Cell] = None
        self.previous_selected_score: float = float("inf")
        self.current_msg_for_distance: Optional[OccupancyGrid] = None

        # Cached planning outputs. These are updated only when planning is allowed
        # to run, but are republished every map update using the latest map header.
        self.cached_candidates: List[FrontierCandidate] = []
        self.cached_selected_candidate: Optional[FrontierCandidate] = None
        self.cached_selected_path_cells: List[Cell] = []
        self.has_planned_once = False
        self.last_plan_time_ns = 0

        self.cached_num_clusters = 0
        self.last_logged_selected_goal_cell: Optional[Cell] = None

        # Robot travel / exploration-progress state.
        self.previous_robot_x: Optional[float] = None
        self.previous_robot_y: Optional[float] = None
        self.previous_robot_yaw: Optional[float] = None
        self.total_robot_distance_m = 0.0
        self.total_abs_yaw_change_rad = 0.0

        self.initial_known_area_m2: Optional[float] = None
        self.initial_free_area_m2: Optional[float] = None
        self.initial_unknown_area_m2: Optional[float] = None
        self.run_start_ros_time_ns: Optional[int] = None

        # Selected-goal lifecycle state.
        self.selected_goal_start_time_ns: Optional[int] = None
        self.selected_goal_start_map_count: Optional[int] = None

        # Rejection/timing diagnostics.
        self.last_candidate_rejection_counts = {
            "clusters_total": 0,
            "clusters_not_evaluated_due_limit": 0,
            "clusters_no_safe_or_reachable_goal": 0,
            "clusters_no_path": 0,
            "clusters_accepted": 0,
        }
        self.current_callback_total_time_ms = 0.0
        self.current_planning_time_ms = 0.0

        self.log_dir: Optional[FilePath] = None
        self.map_metrics_file = None
        self.candidate_metrics_file = None
        self.map_metrics_writer = None
        self.candidate_metrics_writer = None
        self.run_id: Optional[int] = None

        self.setup_logging()

        self.get_logger().info("Frontier detector started.")
        self.get_logger().info(f"Subscribing to: {self.map_topic}")
        self.get_logger().info(f"Robot frame: {self.robot_frame}")
        self.get_logger().info(f"Require reachable: {self.require_reachable}")
        self.get_logger().info(f"Path planning enabled: {self.enable_path_planning}")
        self.get_logger().info(f"Goal hysteresis enabled: {self.enable_goal_hysteresis}")
        self.get_logger().info(
            f"Planning throttle: plan_every_n_maps={self.plan_every_n_maps}, "
            f"min_plan_period_s={self.min_plan_period_s:.2f}"
        )

        if self.enable_logging and self.log_dir is not None:
            self.get_logger().info(f"Logging to: {self.log_dir}")

    def map_callback(self, msg: OccupancyGrid) -> None:
        callback_start_ns = time.perf_counter_ns()
        planning_time_ms = 0.0

        self.map_count += 1
        self.current_msg_for_distance = msg

        width = msg.info.width
        height = msg.info.height

        if width == 0 or height == 0:
            self.get_logger().warn("Received empty occupancy grid.")
            return

        grid = np.asarray(msg.data, dtype=np.int16).reshape((height, width))

        free_mask = (grid >= 0) & (grid <= self.free_max_value)
        unknown_mask = grid == self.unknown_value
        occupied_mask = grid >= self.occupied_min_value

        robot_pose = self.lookup_robot_pose(msg)

        robot_cell = None
        robot_x = None
        robot_y = None
        robot_yaw = None

        if robot_pose is not None:
            robot_cell, robot_x, robot_y, robot_yaw = robot_pose

        raw_frontiers = self.detect_raw_frontiers_numpy(free_mask, unknown_mask)

        goal_safe_mask = self.build_clearance_safe_mask(
            free_mask=free_mask,
            occupied_mask=occupied_mask,
            clearance_m=self.min_obstacle_clearance_m,
            resolution=msg.info.resolution,
        )

        filtered_frontiers = self.filter_frontiers_fast(
            msg=msg,
            frontier_cells=raw_frontiers,
            goal_safe_mask=goal_safe_mask,
            robot_cell=robot_cell,
        )

        # Fast outputs update every map callback.
        self.publish_frontier_grid(msg, filtered_frontiers)

        planning_ran = False

        if self.should_run_planning():
            planning_ran = True
            planning_start_ns = time.perf_counter_ns()

            path_safe_mask = self.build_clearance_safe_mask(
                free_mask=free_mask,
                occupied_mask=occupied_mask,
                clearance_m=self.path_obstacle_clearance_m,
                resolution=msg.info.resolution,
            )

            reachable_cells = None
            if self.require_reachable:
                if robot_cell is None:
                    self.get_logger().warn(
                        "require_reachable=True, but robot pose is unavailable. "
                        "No selected goal/path will be generated for this planning cycle.",
                        throttle_duration_sec=self.robot_pose_warn_period_s,
                    )
                else:
                    reachable_cells = self.compute_reachable_free_cells(
                        msg=msg,
                        robot_cell=robot_cell,
                        path_safe_mask=path_safe_mask,
                    )

            planning_frontiers = self.apply_reachability_to_frontiers(
                frontier_cells=filtered_frontiers,
                reachable_cells=reachable_cells,
            )

            clusters = self.cluster_frontiers(planning_frontiers)
            clusters = [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size_cells]

            candidates = self.build_candidates(
                msg=msg,
                clusters=clusters,
                goal_safe_mask=goal_safe_mask,
                path_safe_mask=path_safe_mask,
                robot_cell=robot_cell,
                reachable_cells=reachable_cells,
            )

            selected_candidate = self.select_candidate(candidates)

            self.cached_candidates = candidates
            self.cached_selected_candidate = selected_candidate
            self.cached_selected_path_cells = (
                selected_candidate.path_cells if selected_candidate is not None else []
            )
            self.cached_num_clusters = len(clusters)

            self.has_planned_once = True
            self.last_plan_time_ns = self.get_clock().now().nanoseconds
            planning_time_ms = (time.perf_counter_ns() - planning_start_ns) * 1e-6

            if self.map_count % self.publish_debug_every_n_maps == 0:
                self.get_logger().info(
                    f"PLAN raw={len(raw_frontiers)}, filtered={len(filtered_frontiers)}, "
                    f"clusters={len(clusters)}, candidates={len(candidates)}"
                )

        selected_goal_pose = (
            self.cached_selected_candidate.goal_pose
            if self.cached_selected_candidate is not None
            else None
        )

        self.publish_goals(msg, [candidate.goal_pose for candidate in self.cached_candidates])
        self.publish_selected_goal(msg, selected_goal_pose)
        self.publish_path(msg, self.cached_selected_path_cells)
        self.publish_markers(
            msg=msg,
            frontier_cells=filtered_frontiers,
            candidates=self.cached_candidates,
            selected_candidate=self.cached_selected_candidate,
            selected_path_cells=self.cached_selected_path_cells,
        )

        self.current_callback_total_time_ms = (time.perf_counter_ns() - callback_start_ns) * 1e-6
        self.current_planning_time_ms = planning_time_ms

        self.log_map_metrics(
            msg=msg,
            free_mask=free_mask,
            unknown_mask=unknown_mask,
            occupied_mask=occupied_mask,
            raw_frontiers=raw_frontiers,
            filtered_frontiers=filtered_frontiers,
            planning_ran=planning_ran,
            robot_cell=robot_cell,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
        )

        if planning_ran:
            self.log_candidate_metrics(
                msg=msg,
                robot_cell=robot_cell,
                robot_x=robot_x,
                robot_y=robot_y,
            )

        if self.map_count % self.publish_debug_every_n_maps == 0 and not planning_ran:
            self.get_logger().info(
                f"FAST raw={len(raw_frontiers)}, filtered={len(filtered_frontiers)}, "
                f"cached_candidates={len(self.cached_candidates)}"
            )

    def should_run_planning(self) -> bool:
        if not self.has_planned_once:
            return True

        if self.plan_every_n_maps <= 1 and self.min_plan_period_s <= 0.0:
            return True

        map_gate_open = True
        if self.plan_every_n_maps > 1:
            map_gate_open = (self.map_count % self.plan_every_n_maps) == 0

        time_gate_open = True
        if self.min_plan_period_s > 0.0:
            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (now_ns - self.last_plan_time_ns) * 1e-9
            time_gate_open = elapsed_s >= self.min_plan_period_s

        return map_gate_open and time_gate_open

    # -------------------------------------------------------------------------
    # CSV logging
    # -------------------------------------------------------------------------

    def setup_logging(self) -> None:
        if not self.enable_logging:
            return

        root = FilePath(self.log_root_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        self.run_id = self.next_run_id(root)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = root / f"run_{self.run_id:03d}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=False)

        self.map_metrics_file = open(
            self.log_dir / "map_metrics.csv",
            mode="w",
            newline="",
        )
        self.candidate_metrics_file = open(
            self.log_dir / "frontier_candidate_metrics.csv",
            mode="w",
            newline="",
        )

        self.map_metrics_writer = csv.DictWriter(
            self.map_metrics_file,
            fieldnames=[
                "run_id",
                "stamp_sec",
                "stamp_nanosec",
                "ros_time_ns",
                "map_count",
                "planning_ran",
                "map_frame",
                "width_cells",
                "height_cells",
                "resolution_m",
                "free_cells",
                "occupied_cells",
                "unknown_cells",
                "known_cells",
                "raw_frontier_cells",
                "filtered_frontier_cells",
                "num_frontier_clusters",
                "num_candidate_goals",
                "free_area_m2",
                "occupied_area_m2",
                "unknown_area_m2",
                "known_area_m2",
                "frontier_length_m",
                "robot_cell_x",
                "robot_cell_y",
                "robot_x",
                "robot_y",
                "robot_yaw",
                "selected_cluster_id",
                "selected_goal_cell_x",
                "selected_goal_cell_y",
                "selected_goal_x",
                "selected_goal_y",
                "selected_goal_yaw",
                "selected_path_length_m",
                "selected_unknown_gain_cells",
                "selected_score",
                "selected_goal_changed",
                "selected_goal_age_s",
                "selected_goal_age_maps",
                "distance_to_selected_goal_m",
                "selection_policy",
                "robot_delta_distance_m",
                "total_robot_distance_m",
                "robot_delta_yaw_rad",
                "total_abs_yaw_change_rad",
                "known_area_gain_m2",
                "free_area_gain_m2",
                "unknown_area_reduction_m2",
                "known_area_gain_rate_m2_per_s",
                "unknown_area_reduction_rate_m2_per_s",
                "known_area_gain_per_m_travelled",
                "unknown_area_reduction_per_m_travelled",
                "clusters_total",
                "clusters_not_evaluated_due_limit",
                "clusters_no_safe_or_reachable_goal",
                "clusters_no_path",
                "clusters_accepted",
                "callback_total_time_ms",
                "planning_time_ms",
            ],
        )

        self.candidate_metrics_writer = csv.DictWriter(
            self.candidate_metrics_file,
            fieldnames=[
                "run_id",
                "stamp_sec",
                "stamp_nanosec",
                "ros_time_ns",
                "map_count",
                "candidate_index",
                "cluster_id",
                "is_selected",
                "goal_cell_x",
                "goal_cell_y",
                "goal_x",
                "goal_y",
                "goal_yaw",
                "path_length_m",
                "euclidean_distance_m",
                "path_to_euclidean_ratio",
                "cluster_size_cells",
                "unknown_gain_cells",
                "unknown_gain_area_m2",
                "unknown_gain_per_path_m",
                "unknown_gain_area_per_path_m",
                "frontier_length_m",
                "bearing_to_goal_rad",
                "heading_error_to_goal_rad",
                "candidate_rank_by_distance",
                "candidate_rank_by_gain",
                "candidate_rank_by_score",
                "score",
                "selection_policy",
            ],
        )

        self.map_metrics_writer.writeheader()
        self.candidate_metrics_writer.writeheader()

        self.map_metrics_file.flush()
        self.candidate_metrics_file.flush()

    def next_run_id(self, root: FilePath) -> int:
        max_id = 0
        pattern = re.compile(r"^run_(\d+)_")

        for child in root.iterdir():
            if not child.is_dir():
                continue

            match = pattern.match(child.name)

            if match is None:
                continue

            max_id = max(max_id, int(match.group(1)))

        return max_id + 1

    def angle_diff(self, a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def update_robot_travel_metrics(
        self,
        robot_x: Optional[float],
        robot_y: Optional[float],
        robot_yaw: Optional[float],
    ) -> Tuple[float, float]:
        if robot_x is None or robot_y is None or robot_yaw is None:
            return 0.0, 0.0

        if (
            self.previous_robot_x is None
            or self.previous_robot_y is None
            or self.previous_robot_yaw is None
        ):
            self.previous_robot_x = robot_x
            self.previous_robot_y = robot_y
            self.previous_robot_yaw = robot_yaw
            return 0.0, 0.0

        dx = robot_x - self.previous_robot_x
        dy = robot_y - self.previous_robot_y
        delta_distance_m = math.hypot(dx, dy)

        delta_yaw_rad = self.angle_diff(robot_yaw, self.previous_robot_yaw)

        self.total_robot_distance_m += delta_distance_m
        self.total_abs_yaw_change_rad += abs(delta_yaw_rad)

        self.previous_robot_x = robot_x
        self.previous_robot_y = robot_y
        self.previous_robot_yaw = robot_yaw

        return delta_distance_m, delta_yaw_rad

    def log_map_metrics(
        self,
        msg: OccupancyGrid,
        free_mask: np.ndarray,
        unknown_mask: np.ndarray,
        occupied_mask: np.ndarray,
        raw_frontiers: Set[Cell],
        filtered_frontiers: Set[Cell],
        planning_ran: bool,
        robot_cell: Optional[Cell],
        robot_x: Optional[float],
        robot_y: Optional[float],
        robot_yaw: Optional[float],
    ) -> None:
        if not self.enable_logging or self.map_metrics_writer is None:
            return

        ros_time_ns = self.get_clock().now().nanoseconds

        free_cells = int(np.count_nonzero(free_mask))
        occupied_cells = int(np.count_nonzero(occupied_mask))
        unknown_cells = int(np.count_nonzero(unknown_mask))
        known_cells = free_cells + occupied_cells

        cell_area_m2 = msg.info.resolution * msg.info.resolution

        free_area_m2 = free_cells * cell_area_m2
        occupied_area_m2 = occupied_cells * cell_area_m2
        unknown_area_m2 = unknown_cells * cell_area_m2
        known_area_m2 = known_cells * cell_area_m2

        if self.run_start_ros_time_ns is None:
            self.run_start_ros_time_ns = ros_time_ns
            self.initial_known_area_m2 = known_area_m2
            self.initial_free_area_m2 = free_area_m2
            self.initial_unknown_area_m2 = unknown_area_m2

        elapsed_s = max((ros_time_ns - self.run_start_ros_time_ns) * 1e-9, 0.0)

        known_area_gain_m2 = known_area_m2 - float(self.initial_known_area_m2)
        free_area_gain_m2 = free_area_m2 - float(self.initial_free_area_m2)
        unknown_area_reduction_m2 = float(self.initial_unknown_area_m2) - unknown_area_m2

        known_area_gain_rate_m2_per_s = ""
        unknown_area_reduction_rate_m2_per_s = ""

        if elapsed_s > 1e-9:
            known_area_gain_rate_m2_per_s = known_area_gain_m2 / elapsed_s
            unknown_area_reduction_rate_m2_per_s = unknown_area_reduction_m2 / elapsed_s

        robot_delta_distance_m, robot_delta_yaw_rad = self.update_robot_travel_metrics(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
        )

        known_area_gain_per_m_travelled = ""
        unknown_area_reduction_per_m_travelled = ""

        if self.total_robot_distance_m > 1e-9:
            known_area_gain_per_m_travelled = known_area_gain_m2 / self.total_robot_distance_m
            unknown_area_reduction_per_m_travelled = (
                unknown_area_reduction_m2 / self.total_robot_distance_m
            )

        selected = self.cached_selected_candidate

        selected_goal_changed = False

        if selected is not None:
            selected_goal_changed = selected.goal_cell != self.last_logged_selected_goal_cell

            if selected_goal_changed or self.selected_goal_start_time_ns is None:
                self.selected_goal_start_time_ns = ros_time_ns
                self.selected_goal_start_map_count = self.map_count

            self.last_logged_selected_goal_cell = selected.goal_cell
        else:
            selected_goal_changed = self.last_logged_selected_goal_cell is not None
            self.last_logged_selected_goal_cell = None
            self.selected_goal_start_time_ns = None
            self.selected_goal_start_map_count = None

        selected_goal_yaw = None
        selected_goal_age_s = ""
        selected_goal_age_maps = ""
        distance_to_selected_goal_m = ""

        if selected is not None:
            selected_goal_yaw = self.pose_yaw(selected.goal_pose)

            if self.selected_goal_start_time_ns is not None:
                selected_goal_age_s = (ros_time_ns - self.selected_goal_start_time_ns) * 1e-9

            if self.selected_goal_start_map_count is not None:
                selected_goal_age_maps = self.map_count - self.selected_goal_start_map_count

            if robot_x is not None and robot_y is not None:
                dx = selected.goal_pose.position.x - robot_x
                dy = selected.goal_pose.position.y - robot_y
                distance_to_selected_goal_m = math.hypot(dx, dy)

        self.map_metrics_writer.writerow(
            {
                "run_id": self.run_id,
                "stamp_sec": msg.header.stamp.sec,
                "stamp_nanosec": msg.header.stamp.nanosec,
                "ros_time_ns": ros_time_ns,
                "map_count": self.map_count,
                "planning_ran": int(planning_ran),
                "map_frame": msg.header.frame_id,
                "width_cells": msg.info.width,
                "height_cells": msg.info.height,
                "resolution_m": msg.info.resolution,
                "free_cells": free_cells,
                "occupied_cells": occupied_cells,
                "unknown_cells": unknown_cells,
                "known_cells": known_cells,
                "raw_frontier_cells": len(raw_frontiers),
                "filtered_frontier_cells": len(filtered_frontiers),
                "num_frontier_clusters": self.cached_num_clusters,
                "num_candidate_goals": len(self.cached_candidates),
                "free_area_m2": free_area_m2,
                "occupied_area_m2": occupied_area_m2,
                "unknown_area_m2": unknown_area_m2,
                "known_area_m2": known_area_m2,
                "frontier_length_m": len(filtered_frontiers) * msg.info.resolution,
                "robot_cell_x": robot_cell[0] if robot_cell is not None else "",
                "robot_cell_y": robot_cell[1] if robot_cell is not None else "",
                "robot_x": robot_x if robot_x is not None else "",
                "robot_y": robot_y if robot_y is not None else "",
                "robot_yaw": robot_yaw if robot_yaw is not None else "",
                "selected_cluster_id": selected.cluster_id if selected is not None else "",
                "selected_goal_cell_x": selected.goal_cell[0] if selected is not None else "",
                "selected_goal_cell_y": selected.goal_cell[1] if selected is not None else "",
                "selected_goal_x": selected.goal_pose.position.x if selected is not None else "",
                "selected_goal_y": selected.goal_pose.position.y if selected is not None else "",
                "selected_goal_yaw": selected_goal_yaw if selected_goal_yaw is not None else "",
                "selected_path_length_m": selected.path_length_m if selected is not None else "",
                "selected_unknown_gain_cells": selected.unknown_gain_cells if selected is not None else "",
                "selected_score": selected.score if selected is not None else "",
                "selected_goal_changed": int(selected_goal_changed),
                "selected_goal_age_s": selected_goal_age_s,
                "selected_goal_age_maps": selected_goal_age_maps,
                "distance_to_selected_goal_m": distance_to_selected_goal_m,
                "selection_policy": self.selection_policy,
                "robot_delta_distance_m": robot_delta_distance_m,
                "total_robot_distance_m": self.total_robot_distance_m,
                "robot_delta_yaw_rad": robot_delta_yaw_rad,
                "total_abs_yaw_change_rad": self.total_abs_yaw_change_rad,
                "known_area_gain_m2": known_area_gain_m2,
                "free_area_gain_m2": free_area_gain_m2,
                "unknown_area_reduction_m2": unknown_area_reduction_m2,
                "known_area_gain_rate_m2_per_s": known_area_gain_rate_m2_per_s,
                "unknown_area_reduction_rate_m2_per_s": unknown_area_reduction_rate_m2_per_s,
                "known_area_gain_per_m_travelled": known_area_gain_per_m_travelled,
                "unknown_area_reduction_per_m_travelled": unknown_area_reduction_per_m_travelled,
                "clusters_total": self.last_candidate_rejection_counts["clusters_total"],
                "clusters_not_evaluated_due_limit": self.last_candidate_rejection_counts[
                    "clusters_not_evaluated_due_limit"
                ],
                "clusters_no_safe_or_reachable_goal": self.last_candidate_rejection_counts[
                    "clusters_no_safe_or_reachable_goal"
                ],
                "clusters_no_path": self.last_candidate_rejection_counts["clusters_no_path"],
                "clusters_accepted": self.last_candidate_rejection_counts["clusters_accepted"],
                "callback_total_time_ms": self.current_callback_total_time_ms,
                "planning_time_ms": self.current_planning_time_ms,
            }
        )

        self.map_metrics_file.flush()

    def log_candidate_metrics(
        self,
        msg: OccupancyGrid,
        robot_cell: Optional[Cell],
        robot_x: Optional[float],
        robot_y: Optional[float],
    ) -> None:
        if not self.enable_logging or self.candidate_metrics_writer is None:
            return

        selected_cell = None
        if self.cached_selected_candidate is not None:
            selected_cell = self.cached_selected_candidate.goal_cell

        cell_area_m2 = msg.info.resolution * msg.info.resolution

        finite_path_candidates = [
            candidate
            for candidate in self.cached_candidates
            if math.isfinite(candidate.path_length_m)
        ]

        rank_by_distance = {
            id(candidate): rank
            for rank, candidate in enumerate(
                sorted(finite_path_candidates, key=lambda item: item.path_length_m),
                start=1,
            )
        }

        rank_by_gain = {
            id(candidate): rank
            for rank, candidate in enumerate(
                sorted(
                    self.cached_candidates,
                    key=lambda item: item.unknown_gain_cells,
                    reverse=True,
                ),
                start=1,
            )
        }

        rank_by_score = {
            id(candidate): rank
            for rank, candidate in enumerate(
                sorted(self.cached_candidates, key=lambda item: item.score),
                start=1,
            )
        }

        for i, candidate in enumerate(self.cached_candidates):
            goal_yaw = self.pose_yaw(candidate.goal_pose)

            euclidean_distance_m = ""
            path_to_euclidean_ratio = ""
            bearing_to_goal_rad = ""
            heading_error_to_goal_rad = ""

            if robot_x is not None and robot_y is not None:
                dx = candidate.goal_pose.position.x - robot_x
                dy = candidate.goal_pose.position.y - robot_y
                euclidean = math.hypot(dx, dy)
                euclidean_distance_m = euclidean

                bearing_to_goal_rad = math.atan2(dy, dx)

                if self.previous_robot_yaw is not None:
                    heading_error_to_goal_rad = self.angle_diff(
                        bearing_to_goal_rad,
                        self.previous_robot_yaw,
                    )

                if euclidean > 1e-9 and math.isfinite(candidate.path_length_m):
                    path_to_euclidean_ratio = candidate.path_length_m / euclidean

            elif robot_cell is not None:
                euclidean = self.cell_distance_m(msg, robot_cell, candidate.goal_cell)
                euclidean_distance_m = euclidean

                if euclidean > 1e-9 and math.isfinite(candidate.path_length_m):
                    path_to_euclidean_ratio = candidate.path_length_m / euclidean

            frontier_length_m = candidate.cluster_size_cells * msg.info.resolution
            unknown_gain_area_m2 = candidate.unknown_gain_cells * cell_area_m2

            unknown_gain_per_path_m = ""
            unknown_gain_area_per_path_m = ""

            if math.isfinite(candidate.path_length_m) and candidate.path_length_m > 1e-9:
                unknown_gain_per_path_m = candidate.unknown_gain_cells / candidate.path_length_m
                unknown_gain_area_per_path_m = unknown_gain_area_m2 / candidate.path_length_m

            self.candidate_metrics_writer.writerow(
                {
                    "run_id": self.run_id,
                    "stamp_sec": msg.header.stamp.sec,
                    "stamp_nanosec": msg.header.stamp.nanosec,
                    "ros_time_ns": self.get_clock().now().nanoseconds,
                    "map_count": self.map_count,
                    "candidate_index": i,
                    "cluster_id": candidate.cluster_id,
                    "is_selected": int(candidate.goal_cell == selected_cell),
                    "goal_cell_x": candidate.goal_cell[0],
                    "goal_cell_y": candidate.goal_cell[1],
                    "goal_x": candidate.goal_pose.position.x,
                    "goal_y": candidate.goal_pose.position.y,
                    "goal_yaw": goal_yaw,
                    "path_length_m": candidate.path_length_m,
                    "euclidean_distance_m": euclidean_distance_m,
                    "path_to_euclidean_ratio": path_to_euclidean_ratio,
                    "cluster_size_cells": candidate.cluster_size_cells,
                    "unknown_gain_cells": candidate.unknown_gain_cells,
                    "unknown_gain_area_m2": unknown_gain_area_m2,
                    "unknown_gain_per_path_m": unknown_gain_per_path_m,
                    "unknown_gain_area_per_path_m": unknown_gain_area_per_path_m,
                    "frontier_length_m": frontier_length_m,
                    "bearing_to_goal_rad": bearing_to_goal_rad,
                    "heading_error_to_goal_rad": heading_error_to_goal_rad,
                    "candidate_rank_by_distance": rank_by_distance.get(id(candidate), ""),
                    "candidate_rank_by_gain": rank_by_gain.get(id(candidate), ""),
                    "candidate_rank_by_score": rank_by_score.get(id(candidate), ""),
                    "score": candidate.score,
                    "selection_policy": self.selection_policy,
                }
            )

        self.candidate_metrics_file.flush()

    def pose_yaw(self, pose: Pose) -> float:
        return self.quaternion_to_yaw(pose.orientation)

    def quaternion_to_yaw(self, q: Quaternion) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def destroy_node(self) -> bool:
        if self.map_metrics_file is not None:
            self.map_metrics_file.flush()
            self.map_metrics_file.close()

        if self.candidate_metrics_file is not None:
            self.candidate_metrics_file.flush()
            self.candidate_metrics_file.close()

        return super().destroy_node()

    # -------------------------------------------------------------------------
    # Occupancy / coordinate helpers
    # -------------------------------------------------------------------------

    def index(self, x: int, y: int, width: int) -> int:
        return y * width + x

    def in_bounds(self, x: int, y: int, width: int, height: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def cell_value(self, msg: OccupancyGrid, x: int, y: int) -> int:
        return msg.data[self.index(x, y, msg.info.width)]

    def is_unknown(self, msg: OccupancyGrid, x: int, y: int) -> bool:
        return self.cell_value(msg, x, y) == self.unknown_value

    def is_free(self, msg: OccupancyGrid, x: int, y: int) -> bool:
        value = self.cell_value(msg, x, y)
        return value >= 0 and value <= self.free_max_value

    def neighbours4(self, x: int, y: int) -> List[Cell]:
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def neighbours8(self, x: int, y: int) -> List[Cell]:
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
        ]

    def frontier_neighbours(self, x: int, y: int) -> List[Cell]:
        return self.neighbours8(x, y) if self.use_8_connected_frontiers else self.neighbours4(x, y)

    def world_to_cell(self, msg: OccupancyGrid, wx: float, wy: float) -> Optional[Cell]:
        cx = int(math.floor((wx - msg.info.origin.position.x) / msg.info.resolution))
        cy = int(math.floor((wy - msg.info.origin.position.y) / msg.info.resolution))

        if not self.in_bounds(cx, cy, msg.info.width, msg.info.height):
            return None

        return (cx, cy)

    def cell_to_world(self, msg: OccupancyGrid, x: int, y: int) -> Tuple[float, float]:
        wx = msg.info.origin.position.x + (x + 0.5) * msg.info.resolution
        wy = msg.info.origin.position.y + (y + 0.5) * msg.info.resolution
        return wx, wy

    def lookup_robot_cell(self, msg: OccupancyGrid) -> Optional[Cell]:
        robot_pose = self.lookup_robot_pose(msg)

        if robot_pose is None:
            return None

        robot_cell, _, _, _ = robot_pose
        return robot_cell

    def lookup_robot_pose(
        self,
        msg: OccupancyGrid,
    ) -> Optional[Tuple[Cell, float, float, float]]:
        map_frame = msg.header.frame_id

        if map_frame == "":
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )

            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            robot_yaw = self.quaternion_to_yaw(transform.transform.rotation)

            robot_cell = self.world_to_cell(msg, robot_x, robot_y)

            if robot_cell is None:
                return None

            return robot_cell, robot_x, robot_y, robot_yaw

        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Fast frontier detection / clearance masks
    # -------------------------------------------------------------------------

    def detect_raw_frontiers_numpy(
        self,
        free_mask: np.ndarray,
        unknown_mask: np.ndarray,
    ) -> Set[Cell]:
        adjacent_unknown = np.zeros_like(unknown_mask, dtype=bool)

        adjacent_unknown[:, 1:] |= unknown_mask[:, :-1]
        adjacent_unknown[:, :-1] |= unknown_mask[:, 1:]
        adjacent_unknown[1:, :] |= unknown_mask[:-1, :]
        adjacent_unknown[:-1, :] |= unknown_mask[1:, :]

        if self.use_8_connected_frontiers:
            adjacent_unknown[1:, 1:] |= unknown_mask[:-1, :-1]
            adjacent_unknown[1:, :-1] |= unknown_mask[:-1, 1:]
            adjacent_unknown[:-1, 1:] |= unknown_mask[1:, :-1]
            adjacent_unknown[:-1, :-1] |= unknown_mask[1:, 1:]

        frontier_mask = free_mask & adjacent_unknown
        ys, xs = np.nonzero(frontier_mask)
        return set(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))

    def build_clearance_safe_mask(
        self,
        free_mask: np.ndarray,
        occupied_mask: np.ndarray,
        clearance_m: float,
        resolution: float,
    ) -> np.ndarray:
        if clearance_m <= 0.0:
            return free_mask.copy()

        radius_cells = int(math.ceil(clearance_m / resolution))

        if radius_cells <= 0:
            return free_mask.copy()

        blocked = occupied_mask.copy()
        height, width = occupied_mask.shape

        offsets: List[Tuple[int, int]] = []
        radius_sq = radius_cells * radius_cells

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_sq:
                    offsets.append((dx, dy))

        occ_y, occ_x = np.nonzero(occupied_mask)

        for dx, dy in offsets:
            shifted_x = occ_x + dx
            shifted_y = occ_y + dy

            valid = (
                (shifted_x >= 0)
                & (shifted_x < width)
                & (shifted_y >= 0)
                & (shifted_y < height)
            )

            blocked[shifted_y[valid], shifted_x[valid]] = True

        return free_mask & (~blocked)

    def filter_frontiers_fast(
        self,
        msg: OccupancyGrid,
        frontier_cells: Set[Cell],
        goal_safe_mask: np.ndarray,
        robot_cell: Optional[Cell],
    ) -> Set[Cell]:
        filtered: Set[Cell] = set()

        for x, y in frontier_cells:
            cell = (x, y)

            if robot_cell is not None:
                if self.cell_distance_m(msg, cell, robot_cell) < self.min_robot_distance_m:
                    continue

            if not goal_safe_mask[y, x]:
                continue

            filtered.add(cell)

        return filtered

    def apply_reachability_to_frontiers(
        self,
        frontier_cells: Set[Cell],
        reachable_cells: Optional[Set[Cell]],
    ) -> Set[Cell]:
        if reachable_cells is None:
            return frontier_cells

        return {cell for cell in frontier_cells if cell in reachable_cells}

    def cell_distance_m(self, msg: OccupancyGrid, a: Cell, b: Cell) -> float:
        return math.hypot(
            (a[0] - b[0]) * msg.info.resolution,
            (a[1] - b[1]) * msg.info.resolution,
        )

    # -------------------------------------------------------------------------
    # Reachability
    # -------------------------------------------------------------------------

    def compute_reachable_free_cells(
        self,
        msg: OccupancyGrid,
        robot_cell: Cell,
        path_safe_mask: np.ndarray,
    ) -> Set[Cell]:
        width = msg.info.width
        height = msg.info.height

        if not self.in_bounds(robot_cell[0], robot_cell[1], width, height):
            return set()

        if not path_safe_mask[robot_cell[1], robot_cell[0]]:
            return set()

        reachable: Set[Cell] = {robot_cell}
        queue = deque([robot_cell])

        while queue:
            x, y = queue.popleft()

            for nx, ny in self.neighbours4(x, y):
                if not self.in_bounds(nx, ny, width, height):
                    continue

                neighbour = (nx, ny)

                if neighbour in reachable:
                    continue

                if not path_safe_mask[ny, nx]:
                    continue

                reachable.add(neighbour)
                queue.append(neighbour)

        return reachable

    # -------------------------------------------------------------------------
    # Clustering
    # -------------------------------------------------------------------------

    def cluster_frontiers(self, frontier_cells: Set[Cell]) -> List[List[Cell]]:
        unvisited = set(frontier_cells)
        clusters: List[List[Cell]] = []

        while unvisited:
            start = unvisited.pop()
            cluster = [start]
            queue = deque([start])

            while queue:
                x, y = queue.popleft()

                for neighbour in self.neighbours8(x, y):
                    if neighbour not in unvisited:
                        continue

                    unvisited.remove(neighbour)
                    queue.append(neighbour)
                    cluster.append(neighbour)

            clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        return clusters

    # -------------------------------------------------------------------------
    # Candidate generation
    # -------------------------------------------------------------------------

    def build_candidates(
        self,
        msg: OccupancyGrid,
        clusters: List[List[Cell]],
        goal_safe_mask: np.ndarray,
        path_safe_mask: np.ndarray,
        robot_cell: Optional[Cell],
        reachable_cells: Optional[Set[Cell]],
    ) -> List[FrontierCandidate]:
        candidates: List[FrontierCandidate] = []

        self.last_candidate_rejection_counts = {
            "clusters_total": len(clusters),
            "clusters_not_evaluated_due_limit": 0,
            "clusters_no_safe_or_reachable_goal": 0,
            "clusters_no_path": 0,
            "clusters_accepted": 0,
        }

        for cluster_id, cluster in enumerate(clusters):
            if len(candidates) >= self.max_goals_to_publish:
                self.last_candidate_rejection_counts[
                    "clusters_not_evaluated_due_limit"
                ] = max(len(clusters) - cluster_id, 0)
                break

            goal_cell = self.find_goal_cell_for_cluster(
                msg=msg,
                cluster=cluster,
                goal_safe_mask=goal_safe_mask,
                reachable_cells=reachable_cells,
            )

            if goal_cell is None:
                self.last_candidate_rejection_counts[
                    "clusters_no_safe_or_reachable_goal"
                ] += 1
                continue

            goal_pose = self.cell_to_pose_facing_unknown(msg, goal_cell, cluster)
            unknown_gain_cells = self.estimate_unknown_gain_cells(msg, cluster)

            path_cells: List[Cell] = []
            path_length_m = float("inf")

            if self.enable_path_planning and robot_cell is not None:
                path_cells, path_length_m = self.astar(
                    msg=msg,
                    start=robot_cell,
                    goal=goal_cell,
                    path_safe_mask=path_safe_mask,
                )

                if not path_cells:
                    self.last_candidate_rejection_counts["clusters_no_path"] += 1
                    continue

            candidate = FrontierCandidate(
                cluster_id=cluster_id,
                goal_cell=goal_cell,
                goal_pose=goal_pose,
                path_cells=path_cells,
                path_length_m=path_length_m,
                cluster_size_cells=len(cluster),
                unknown_gain_cells=unknown_gain_cells,
            )

            candidates.append(candidate)
            self.last_candidate_rejection_counts["clusters_accepted"] += 1

        return candidates

    def estimate_unknown_gain_cells(self, msg: OccupancyGrid, cluster: List[Cell]) -> int:
        unknown_cells: Set[Cell] = set()

        for fx, fy in cluster:
            for nx, ny in self.frontier_neighbours(fx, fy):
                if not self.in_bounds(nx, ny, msg.info.width, msg.info.height):
                    continue

                if self.is_unknown(msg, nx, ny):
                    unknown_cells.add((nx, ny))

        return len(unknown_cells)

    def find_goal_cell_for_cluster(
        self,
        msg: OccupancyGrid,
        cluster: List[Cell],
        goal_safe_mask: np.ndarray,
        reachable_cells: Optional[Set[Cell]],
    ) -> Optional[Cell]:
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        cx = sum(cell[0] for cell in cluster) / float(len(cluster))
        cy = sum(cell[1] for cell in cluster) / float(len(cluster))

        centre_x = int(round(cx))
        centre_y = int(round(cy))

        search_radius_cells = int(math.ceil(self.goal_search_radius_m / resolution))

        candidates: List[Tuple[float, Cell]] = []

        for y in range(centre_y - search_radius_cells, centre_y + search_radius_cells + 1):
            for x in range(centre_x - search_radius_cells, centre_x + search_radius_cells + 1):
                if not self.in_bounds(x, y, width, height):
                    continue

                if not goal_safe_mask[y, x]:
                    continue

                cell = (x, y)

                if reachable_cells is not None and cell not in reachable_cells:
                    continue

                dist_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                candidates.append((dist_sq, cell))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def cell_to_pose_facing_unknown(
        self,
        msg: OccupancyGrid,
        goal_cell: Cell,
        cluster: List[Cell],
    ) -> Pose:
        gx, gy = goal_cell
        wx, wy = self.cell_to_world(msg, gx, gy)

        unknown_vectors: List[Tuple[float, float]] = []

        for fx, fy in cluster:
            for nx, ny in self.frontier_neighbours(fx, fy):
                if not self.in_bounds(nx, ny, msg.info.width, msg.info.height):
                    continue

                if self.is_unknown(msg, nx, ny):
                    unknown_vectors.append((float(nx - gx), float(ny - gy)))

        if unknown_vectors:
            vx = sum(v[0] for v in unknown_vectors) / float(len(unknown_vectors))
            vy = sum(v[1] for v in unknown_vectors) / float(len(unknown_vectors))
            yaw = math.atan2(vy, vx)
        else:
            yaw = 0.0

        pose = Pose()
        pose.position.x = wx
        pose.position.y = wy
        pose.position.z = 0.0
        pose.orientation = self.yaw_to_quaternion(yaw)
        return pose

    def yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def select_candidate(self, candidates: List[FrontierCandidate]) -> Optional[FrontierCandidate]:
        if not candidates:
            self.previous_selected_goal_cell = None
            self.previous_selected_score = float("inf")
            return None

        self.compute_candidate_scores(candidates)
        best_candidate = min(candidates, key=lambda candidate: candidate.score)

        if not self.enable_goal_hysteresis:
            self.update_hysteresis_state(best_candidate)
            return best_candidate

        previous_candidate = self.find_previous_selected_candidate(candidates)

        if previous_candidate is None:
            self.update_hysteresis_state(best_candidate)
            return best_candidate

        should_switch = best_candidate.score < previous_candidate.score - self.hysteresis_switch_margin
        selected = best_candidate if should_switch else previous_candidate

        self.update_hysteresis_state(selected)
        return selected

    def compute_candidate_scores(self, candidates: List[FrontierCandidate]) -> None:
        if self.selection_policy == "nearest":
            for candidate in candidates:
                candidate.score = candidate.path_length_m
            return

        if self.selection_policy == "utility":
            for candidate in candidates:
                candidate.score = (
                    self.utility_distance_weight * candidate.path_length_m
                    - self.utility_gain_weight * float(candidate.unknown_gain_cells)
                )
            return

        self.get_logger().warn(
            f'Unknown selection_policy="{self.selection_policy}". Falling back to nearest.'
        )

        for candidate in candidates:
            candidate.score = candidate.path_length_m

    def find_previous_selected_candidate(
        self,
        candidates: List[FrontierCandidate],
    ) -> Optional[FrontierCandidate]:
        if self.previous_selected_goal_cell is None:
            return None

        if self.current_msg_for_distance is None:
            return None

        best_match = None
        best_distance = float("inf")

        for candidate in candidates:
            distance = self.cell_distance_m(
                self.current_msg_for_distance,
                candidate.goal_cell,
                self.previous_selected_goal_cell,
            )

            if distance < best_distance:
                best_distance = distance
                best_match = candidate

        if best_match is None:
            return None

        if best_distance > self.hysteresis_goal_match_distance_m:
            return None

        return best_match

    def update_hysteresis_state(self, selected_candidate: FrontierCandidate) -> None:
        self.previous_selected_goal_cell = selected_candidate.goal_cell
        self.previous_selected_score = selected_candidate.score

    # -------------------------------------------------------------------------
    # A*
    # -------------------------------------------------------------------------

    def astar(
        self,
        msg: OccupancyGrid,
        start: Cell,
        goal: Cell,
        path_safe_mask: np.ndarray,
    ) -> Tuple[List[Cell], float]:
        width = msg.info.width
        height = msg.info.height

        if not self.in_bounds(start[0], start[1], width, height):
            return [], float("inf")

        if not self.in_bounds(goal[0], goal[1], width, height):
            return [], float("inf")

        if not path_safe_mask[start[1], start[0]]:
            return [], float("inf")

        if not path_safe_mask[goal[1], goal[0]]:
            return [], float("inf")

        open_heap: List[Tuple[float, float, Cell]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))

        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}
        closed: Set[Cell] = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return path, g_score[current] * msg.info.resolution

            closed.add(current)

            for neighbour, step_cost in self.astar_neighbours(
                msg=msg,
                cell=current,
                path_safe_mask=path_safe_mask,
            ):
                if neighbour in closed:
                    continue

                tentative_g = current_g + step_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    priority = tentative_g + self.heuristic_cells(neighbour, goal)
                    heapq.heappush(open_heap, (priority, tentative_g, neighbour))

        return [], float("inf")

    def astar_neighbours(
        self,
        msg: OccupancyGrid,
        cell: Cell,
        path_safe_mask: np.ndarray,
    ) -> List[Tuple[Cell, float]]:
        x, y = cell

        if self.allow_diagonal_motion:
            candidate_moves = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 1, math.sqrt(2.0)),
                (1, -1, math.sqrt(2.0)),
                (-1, 1, math.sqrt(2.0)),
                (-1, -1, math.sqrt(2.0)),
            ]
        else:
            candidate_moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]

        neighbours: List[Tuple[Cell, float]] = []

        for dx, dy, cost in candidate_moves:
            nx = x + dx
            ny = y + dy

            if not self.in_bounds(nx, ny, msg.info.width, msg.info.height):
                continue

            if not path_safe_mask[ny, nx]:
                continue

            is_diagonal = dx != 0 and dy != 0

            if is_diagonal and self.prevent_diagonal_corner_cutting:
                if not path_safe_mask[y, nx]:
                    continue
                if not path_safe_mask[ny, x]:
                    continue

            neighbours.append(((nx, ny), cost))

        return neighbours

    def heuristic_cells(self, a: Cell, b: Cell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reconstruct_path(self, came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    # -------------------------------------------------------------------------
    # Publishers
    # -------------------------------------------------------------------------

    def publish_frontier_grid(self, msg: OccupancyGrid, frontier_cells: Set[Cell]) -> None:
        data = np.zeros((msg.info.height, msg.info.width), dtype=np.int8)

        for x, y in frontier_cells:
            data[y, x] = 100

        frontier_grid = OccupancyGrid()
        frontier_grid.header = msg.header
        frontier_grid.info = msg.info
        frontier_grid.data = data.reshape(-1).astype(int).tolist()

        self.frontier_grid_pub.publish(frontier_grid)

    def publish_goals(self, msg: OccupancyGrid, goals: List[Pose]) -> None:
        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.poses = goals
        self.goal_pub.publish(pose_array)

    def publish_selected_goal(self, msg: OccupancyGrid, selected_goal: Optional[Pose]) -> None:
        if selected_goal is None:
            return

        stamped = PoseStamped()
        stamped.header = msg.header
        stamped.pose = selected_goal
        self.selected_goal_pub.publish(stamped)

    def publish_path(self, msg: OccupancyGrid, path_cells: List[Cell]) -> None:
        path = Path()
        path.header = msg.header

        for x, y in path_cells:
            wx, wy = self.cell_to_world(msg, x, y)

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.05
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_pub.publish(path)

    def publish_markers(
        self,
        msg: OccupancyGrid,
        frontier_cells: Set[Cell],
        candidates: List[FrontierCandidate],
        selected_candidate: Optional[FrontierCandidate],
        selected_path_cells: List[Cell],
    ) -> None:
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header = msg.header
        delete_marker.ns = "frontier_detector"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        frontier_points = Marker()
        frontier_points.header = msg.header
        frontier_points.ns = "frontier_cells"
        frontier_points.id = 1
        frontier_points.type = Marker.POINTS
        frontier_points.action = Marker.ADD
        frontier_points.scale.x = self.marker_scale_m
        frontier_points.scale.y = self.marker_scale_m
        frontier_points.color.r = 1.0
        frontier_points.color.g = 0.35
        frontier_points.color.b = 0.0
        frontier_points.color.a = 1.0

        for x, y in frontier_cells:
            wx, wy = self.cell_to_world(msg, x, y)
            point = Point()
            point.x = wx
            point.y = wy
            point.z = 0.05
            frontier_points.points.append(point)

        marker_array.markers.append(frontier_points)

        for i, candidate in enumerate(candidates):
            sphere = Marker()
            sphere.header = msg.header
            sphere.ns = "candidate_frontier_goals"
            sphere.id = 1000 + i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = candidate.goal_pose
            sphere.pose.position.z = 0.15
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25
            sphere.color.r = 0.0
            sphere.color.g = 0.8
            sphere.color.b = 1.0
            sphere.color.a = 0.9
            marker_array.markers.append(sphere)

            if self.publish_text_labels:
                text = Marker()
                text.header = msg.header
                text.ns = "frontier_goal_labels"
                text.id = 2000 + i
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose = candidate.goal_pose
                text.pose.position.z = 0.5
                text.scale.z = 0.22
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = 1.0
                text.text = (
                    f"F{candidate.cluster_id}\\n"
                    f"{candidate.path_length_m:.1f}m\\n"
                    f"g={candidate.unknown_gain_cells}"
                )
                marker_array.markers.append(text)

        if selected_candidate is not None:
            selected = Marker()
            selected.header = msg.header
            selected.ns = "selected_frontier_goal"
            selected.id = 3000
            selected.type = Marker.SPHERE
            selected.action = Marker.ADD
            selected.pose = selected_candidate.goal_pose
            selected.pose.position.z = 0.25
            selected.scale.x = 0.45
            selected.scale.y = 0.45
            selected.scale.z = 0.45
            selected.color.r = 1.0
            selected.color.g = 0.0
            selected.color.b = 1.0
            selected.color.a = 1.0
            marker_array.markers.append(selected)

        if selected_path_cells:
            path_marker = Marker()
            path_marker.header = msg.header
            path_marker.ns = "selected_frontier_path"
            path_marker.id = 4000
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.06
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0

            for x, y in selected_path_cells:
                wx, wy = self.cell_to_world(msg, x, y)
                point = Point()
                point.x = wx
                point.y = wy
                point.z = 0.12
                path_marker.points.append(point)

            marker_array.markers.append(path_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
