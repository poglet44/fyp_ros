#!/usr/bin/env python3

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class RLTrainingEnvStub(Node):
    """
    Sim-only RL training environment stub.

    This validates the RL transition loop:

      observe candidates
      choose action
      publish selected goal/path
      wait for action outcome
      compute reward
      log transition

    The current dummy policy chooses the nearest candidate from the same
    region-balanced top-N set intended for RL.
    """

    RL_FEATURE_KEYS = [
        "path_length_norm",
        "euclidean_distance_norm",
        "path_to_euclidean_ratio_norm",
        "frontier_length_norm",
        "local_unknown_neighbor_area_norm",
        "local_unknown_area_per_path_norm",
        "local_information_density_norm",
        "bearing_to_goal_sin",
        "bearing_to_goal_cos",
        "heading_error_sin",
        "heading_error_cos",
        "stability_cycles_norm",
    ]

    RECOVERY_STATES = {
        "RECOVERY_CHECKPOINT",
        "RECOVERY_RETURN_START",
        "RECOVERY_LOCAL_RECHECK",
    }

    def __init__(self):
        super().__init__("rl_training_env_stub")

        # Topics
        self.declare_parameter("candidates_topic", "/frontier_candidates")
        self.declare_parameter("exploration_grid_topic", "/exploration_grid")
        self.declare_parameter("supervisor_status_topic", "/exploration_supervisor/status")
        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal_external")
        self.declare_parameter("selected_path_topic", "/selected_frontier_path_external")
        self.declare_parameter("status_topic", "/rl_training_env_stub/status")

        # Frames
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("tf_lookup_timeout_s", 0.20)

        # RL/action settings
        self.declare_parameter("max_rl_candidates", 10)
        self.declare_parameter("goal_reached_distance_m", 0.75)
        self.declare_parameter("active_goal_match_radius_m", 0.50)
        self.declare_parameter("max_action_duration_s", 120.0)
        self.declare_parameter("min_action_timeout_s", 30.0)
        self.declare_parameter("expected_navigation_speed_mps", 0.25)
        self.declare_parameter("action_timeout_scale", 2.5)
        self.declare_parameter("goal_invalid_grace_s", 5.0)
        self.declare_parameter("min_action_duration_s", 8.0)
        self.declare_parameter("min_goal_distance_reduction_m", 0.5)  # Deprecated for poor-progress logic.
        self.declare_parameter("min_known_area_gain_m2", 1.0)
        self.declare_parameter("min_robot_motion_m", 0.75)
        self.declare_parameter("failed_goal_cooldown_s", 45.0)
        self.declare_parameter("failed_goal_radius_m", 1.0)
        self.declare_parameter("publish_rate_hz", 5.0)

        # Reward
        self.declare_parameter("distance_penalty_lambda", 0.10)
        self.declare_parameter("recovery_penalty", 2.0)
        self.declare_parameter("timeout_penalty", 2.0)
        self.declare_parameter("poor_progress_penalty", 2.0)
        self.declare_parameter("invalid_action_penalty", 5.0)

        # Logging
        self.declare_parameter("log_root_dir", "~/Sam/fyp_ws/logs/rl_training_stub")
        self.declare_parameter("episode_name", "manual_stub_episode")

        self.candidates_topic = str(self.get_parameter("candidates_topic").value)
        self.exploration_grid_topic = str(self.get_parameter("exploration_grid_topic").value)
        self.supervisor_status_topic = str(self.get_parameter("supervisor_status_topic").value)
        self.selected_goal_topic = str(self.get_parameter("selected_goal_topic").value)
        self.selected_path_topic = str(self.get_parameter("selected_path_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)

        self.max_rl_candidates = int(self.get_parameter("max_rl_candidates").value)
        self.max_rl_candidates = max(1, self.max_rl_candidates)
        self.goal_reached_distance_m = float(self.get_parameter("goal_reached_distance_m").value)
        self.active_goal_match_radius_m = float(self.get_parameter("active_goal_match_radius_m").value)
        self.max_action_duration_s = float(self.get_parameter("max_action_duration_s").value)
        self.min_action_timeout_s = float(self.get_parameter("min_action_timeout_s").value)
        self.expected_navigation_speed_mps = float(self.get_parameter("expected_navigation_speed_mps").value)
        self.expected_navigation_speed_mps = max(0.01, self.expected_navigation_speed_mps)
        self.action_timeout_scale = float(self.get_parameter("action_timeout_scale").value)
        self.goal_invalid_grace_s = float(self.get_parameter("goal_invalid_grace_s").value)
        self.min_action_duration_s = float(self.get_parameter("min_action_duration_s").value)
        self.min_goal_distance_reduction_m = float(self.get_parameter("min_goal_distance_reduction_m").value)
        self.min_known_area_gain_m2 = float(self.get_parameter("min_known_area_gain_m2").value)
        self.min_robot_motion_m = float(self.get_parameter("min_robot_motion_m").value)
        self.failed_goal_cooldown_s = float(self.get_parameter("failed_goal_cooldown_s").value)
        self.failed_goal_radius_m = float(self.get_parameter("failed_goal_radius_m").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.distance_penalty_lambda = float(self.get_parameter("distance_penalty_lambda").value)
        self.recovery_penalty = float(self.get_parameter("recovery_penalty").value)
        self.timeout_penalty = float(self.get_parameter("timeout_penalty").value)
        self.poor_progress_penalty = float(self.get_parameter("poor_progress_penalty").value)
        self.invalid_action_penalty = float(self.get_parameter("invalid_action_penalty").value)

        self.log_root_dir = Path(str(self.get_parameter("log_root_dir").value)).expanduser()
        self.episode_name = str(self.get_parameter("episode_name").value)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(String, self.candidates_topic, self.candidates_callback, 10)
        self.create_subscription(OccupancyGrid, self.exploration_grid_topic, self.grid_callback, 10)
        self.create_subscription(String, self.supervisor_status_topic, self.supervisor_status_callback, 10)

        self.selected_goal_pub = self.create_publisher(PoseStamped, self.selected_goal_topic, 10)
        self.selected_path_pub = self.create_publisher(NavPath, self.selected_path_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        timer_period = 1.0 / max(self.publish_rate_hz, 0.1)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.latest_candidates_payload: Optional[Dict[str, Any]] = None
        self.latest_grid: Optional[OccupancyGrid] = None
        self.latest_supervisor_status: Dict[str, Any] = {}

        self.active_action = False
        self.active_goal: Optional[PoseStamped] = None
        self.active_path: Optional[NavPath] = None
        self.active_candidate: Optional[Dict[str, Any]] = None
        self.active_action_index: Optional[int] = None
        self.active_preselected_indices: List[int] = []
        self.active_valid_action_mask: List[int] = []

        self.action_start_time_s: Optional[float] = None
        self.active_action_timeout_s: Optional[float] = None
        self.action_start_known_area_m2: Optional[float] = None
        self.action_start_robot_xy: Optional[Tuple[float, float]] = None
        self.action_start_goal_distance_m: Optional[float] = None
        self.last_robot_xy: Optional[Tuple[float, float]] = None
        self.action_distance_m = 0.0
        self.active_candidate_invalid_since_s: Optional[float] = None

        self.transition_count = 0
        self.failed_goal_cooldowns: List[Dict[str, Any]] = []

        self.setup_logging()

        self.get_logger().info("RL training env stub started.")
        self.get_logger().info(f"Publishing selected goal: {self.selected_goal_topic}")
        self.get_logger().info(f"Publishing selected path: {self.selected_path_topic}")
        self.get_logger().info(f"Logging transitions to: {self.transitions_csv_path}")

    def setup_logging(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.log_root_dir / f"{self.episode_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.transitions_csv_path = self.run_dir / "transitions.csv"
        self.observations_jsonl_path = self.run_dir / "observations.jsonl"

        self.transitions_file = self.transitions_csv_path.open("w", newline="")
        self.transitions_writer = csv.DictWriter(
            self.transitions_file,
            fieldnames=[
                "transition_index",
                "start_time_s",
                "end_time_s",
                "duration_s",
                "action_index",
                "candidate_index",
                "cluster_id",
                "region_id",
                "goal_x",
                "goal_y",
                "path_length_m",
                "known_area_before_m2",
                "known_area_after_m2",
                "known_area_delta_m2",
                "distance_travelled_m",
                "distance_penalty_lambda",
                "failure_penalty",
                "reward",
                "end_reason",
                "preselected_candidate_indices",
                "valid_action_mask",
                "failed_goal_cooldown_count",
            ],
        )
        self.transitions_writer.writeheader()
        self.transitions_file.flush()

        self.observations_file = self.observations_jsonl_path.open("w")

    def candidates_callback(self, msg: String) -> None:
        try:
            self.latest_candidates_payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Ignoring invalid /frontier_candidates JSON.")

    def grid_callback(self, msg: OccupancyGrid) -> None:
        self.latest_grid = msg

    def supervisor_status_callback(self, msg: String) -> None:
        try:
            self.latest_supervisor_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.latest_supervisor_status = {}

    def timer_callback(self) -> None:
        self.update_active_distance()

        if not self.active_action:
            self.try_start_action()
        else:
            self.republish_active_goal_and_path()
            self.check_action_end()

        self.publish_status()

    def try_start_action(self) -> None:
        if self.latest_candidates_payload is None or self.latest_grid is None:
            return

        supervisor_state = str(self.latest_supervisor_status.get("state", ""))

        if supervisor_state != "EXPLORING":
            return

        candidates = self.latest_candidates_payload.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return

        preselected = self.preselect_rl_candidates(candidates)
        valid_action_mask = [1] * len(preselected) + [0] * max(0, self.max_rl_candidates - len(preselected))

        if not preselected:
            return

        observation = self.build_rl_observation(preselected)

        # Dummy policy for transition validation:
        # choose nearest candidate from the RL preselected set.
        action_index = self.choose_nearest_action(preselected)

        if action_index is None or action_index >= len(preselected):
            self.log_invalid_action(
                observation=observation,
                valid_action_mask=valid_action_mask,
                action_index=-1 if action_index is None else action_index,
            )
            return

        selected_candidate = preselected[action_index]

        goal = self.candidate_to_pose_stamped(self.latest_candidates_payload, selected_candidate)
        path = self.candidate_to_path(self.latest_candidates_payload, selected_candidate)

        known_area = self.compute_known_area_m2(self.latest_grid)
        robot_xy = self.lookup_robot_xy()

        self.active_action = True
        self.active_goal = goal
        self.active_path = path
        self.active_candidate = selected_candidate
        self.active_action_index = action_index
        self.active_preselected_indices = [
            int(candidate.get("candidate_index", -1))
            for candidate in preselected
        ]
        self.active_valid_action_mask = valid_action_mask

        self.action_start_time_s = self.now_seconds()
        self.active_action_timeout_s = self.compute_action_timeout_s(selected_candidate)
        self.action_start_known_area_m2 = known_area
        self.action_start_robot_xy = robot_xy
        self.action_start_goal_distance_m = self.distance_robot_to_goal(robot_xy, goal)
        self.last_robot_xy = robot_xy
        self.action_distance_m = 0.0
        self.active_candidate_invalid_since_s = None

        self.write_observation_jsonl(
            observation=observation,
            valid_action_mask=valid_action_mask,
            preselected=preselected,
            action_index=action_index,
        )

        self.republish_active_goal_and_path()

        self.get_logger().info(
            f"Started action {action_index}: "
            f"candidate_index={selected_candidate.get('candidate_index')}, "
            f"path_length={selected_candidate.get('path_length_m')}"
        )

    def compute_action_timeout_s(self, candidate: Dict[str, Any]) -> float:
        """
        Compute a path-length-scaled action timeout.

        This avoids treating long valid goals as failures simply because they
        require more travel time.
        """
        path_length_m = self.finite_path_length(candidate)

        if not math.isfinite(path_length_m):
            return self.max_action_duration_s

        expected_time_s = path_length_m / self.expected_navigation_speed_mps
        scaled_timeout_s = expected_time_s * self.action_timeout_scale

        return max(
            self.min_action_timeout_s,
            min(self.max_action_duration_s, scaled_timeout_s),
        )

    def check_action_end(self) -> None:
        if not self.active_action:
            return

        now_s = self.now_seconds()
        elapsed_s = now_s - float(self.action_start_time_s)

        supervisor_state = str(self.latest_supervisor_status.get("state", ""))

        if supervisor_state in self.RECOVERY_STATES:
            self.finish_action("recovery_triggered")
            return

        if self.goal_reached():
            self.finish_action("goal_reached")
            return

        if self.active_candidate_invalid_for_too_long():
            self.finish_action("goal_invalidated")
            return

        if self.poor_progress_detected(elapsed_s):
            self.finish_action("poor_progress")
            return

        timeout_s = (
            self.active_action_timeout_s
            if self.active_action_timeout_s is not None
            else self.max_action_duration_s
        )

        if elapsed_s >= timeout_s:
            self.finish_action("goal_timeout")
            return

    def finish_action(self, end_reason: str) -> None:
        if self.latest_grid is None:
            return

        end_time_s = self.now_seconds()
        duration_s = end_time_s - float(self.action_start_time_s)

        known_before = float(self.action_start_known_area_m2)
        known_after = self.compute_known_area_m2(self.latest_grid)
        known_delta = known_after - known_before

        failure_penalty = 0.0
        if end_reason == "recovery_triggered":
            failure_penalty = self.recovery_penalty
        elif end_reason == "goal_timeout":
            failure_penalty = self.timeout_penalty
        elif end_reason == "goal_invalidated":
            failure_penalty = self.timeout_penalty
        elif end_reason == "poor_progress":
            failure_penalty = self.poor_progress_penalty

        reward = (
            known_delta
            - self.distance_penalty_lambda * self.action_distance_m
            - failure_penalty
        )

        c = self.active_candidate or {}

        if end_reason in {"recovery_triggered", "goal_timeout"}:
            self.add_failed_goal_cooldown(end_reason=end_reason)

        self.transitions_writer.writerow(
            {
                "transition_index": self.transition_count,
                "start_time_s": self.action_start_time_s,
                "end_time_s": end_time_s,
                "duration_s": duration_s,
                "action_index": self.active_action_index,
                "candidate_index": c.get("candidate_index"),
                "cluster_id": c.get("cluster_id"),
                "region_id": c.get("region_id"),
                "goal_x": c.get("goal_x"),
                "goal_y": c.get("goal_y"),
                "path_length_m": c.get("path_length_m"),
                "known_area_before_m2": known_before,
                "known_area_after_m2": known_after,
                "known_area_delta_m2": known_delta,
                "distance_travelled_m": self.action_distance_m,
                "distance_penalty_lambda": self.distance_penalty_lambda,
                "failure_penalty": failure_penalty,
                "reward": reward,
                "end_reason": end_reason,
                "preselected_candidate_indices": ";".join(str(x) for x in self.active_preselected_indices),
                "valid_action_mask": ";".join(str(x) for x in self.active_valid_action_mask),
                "failed_goal_cooldown_count": len(self.failed_goal_cooldowns),
            }
        )
        self.transitions_file.flush()

        self.get_logger().info(
            f"Finished action: reason={end_reason}, "
            f"known_delta={known_delta:.3f}, "
            f"distance={self.action_distance_m:.3f}, "
            f"reward={reward:.3f}"
        )

        self.transition_count += 1
        self.clear_active_action()

    def clear_active_action(self) -> None:
        self.active_action = False
        self.active_goal = None
        self.active_path = None
        self.active_candidate = None
        self.active_action_index = None
        self.active_preselected_indices = []
        self.active_valid_action_mask = []
        self.action_start_time_s = None
        self.active_action_timeout_s = None
        self.action_start_known_area_m2 = None
        self.action_start_robot_xy = None
        self.action_start_goal_distance_m = None
        self.last_robot_xy = None
        self.action_distance_m = 0.0
        self.active_candidate_invalid_since_s = None

    def republish_active_goal_and_path(self) -> None:
        if self.active_goal is None or self.active_path is None:
            return

        now_msg = self.get_clock().now().to_msg()

        self.active_goal.header.stamp = now_msg
        self.active_path.header.stamp = now_msg

        for pose in self.active_path.poses:
            pose.header.stamp = now_msg

        self.selected_goal_pub.publish(self.active_goal)
        self.selected_path_pub.publish(self.active_path)

    def update_active_distance(self) -> None:
        if not self.active_action:
            return

        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            return

        if self.last_robot_xy is not None:
            dx = robot_xy[0] - self.last_robot_xy[0]
            dy = robot_xy[1] - self.last_robot_xy[1]
            self.action_distance_m += math.hypot(dx, dy)

        self.last_robot_xy = robot_xy

    def distance_robot_to_goal(
        self,
        robot_xy: Optional[Tuple[float, float]],
        goal: Optional[PoseStamped],
    ) -> Optional[float]:
        if robot_xy is None or goal is None:
            return None

        return math.hypot(
            goal.pose.position.x - robot_xy[0],
            goal.pose.position.y - robot_xy[1],
        )

    def poor_progress_detected(self, elapsed_s: float) -> bool:
        """
        Detect poor action progress.

        This deliberately uses actual robot motion and actual known-area gain,
        not distance-to-goal reduction. Distance-to-goal is unreliable here
        because the active frontier goal may be refreshed/moved as the map
        updates.

        Poor progress means:
          after min_action_duration_s,
          the robot has gained little known area,
          and the robot has barely physically moved.
        """
        if elapsed_s < self.min_action_duration_s:
            return False

        if self.latest_grid is None:
            return False

        if self.action_start_known_area_m2 is None:
            return False

        known_now = self.compute_known_area_m2(self.latest_grid)
        known_gain = known_now - float(self.action_start_known_area_m2)

        if known_gain >= self.min_known_area_gain_m2:
            return False

        if self.action_distance_m >= self.min_robot_motion_m:
            return False

        return True

    def goal_reached(self) -> bool:
        if self.active_goal is None:
            return False

        robot_xy = self.lookup_robot_xy()
        if robot_xy is None:
            return False

        gx = self.active_goal.pose.position.x
        gy = self.active_goal.pose.position.y

        return math.hypot(gx - robot_xy[0], gy - robot_xy[1]) <= self.goal_reached_distance_m

    def find_current_match_for_active_candidate(self) -> Optional[Dict[str, Any]]:
        """
        Find the current live frontier candidate that best corresponds to the
        active RL-selected target.

        Priority:
          1. Same region_id and close to the current active goal.
          2. Any candidate close to the current active goal.

        This prevents the robot from continuing toward a stale coordinate when
        the frontier shifts as the map updates.
        """
        if (
            self.active_candidate is None
            or self.active_goal is None
            or self.latest_candidates_payload is None
        ):
            return None

        candidates = self.latest_candidates_payload.get("candidates", [])
        if not isinstance(candidates, list):
            return None

        active_x = float(self.active_goal.pose.position.x)
        active_y = float(self.active_goal.pose.position.y)

        try:
            active_region_id = int(self.active_candidate.get("region_id", -999999))
        except (TypeError, ValueError):
            active_region_id = -999999

        same_region_matches = []
        any_region_matches = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            try:
                candidate_x = float(candidate.get("goal_x"))
                candidate_y = float(candidate.get("goal_y"))
            except (TypeError, ValueError):
                continue

            distance_to_active_goal = math.hypot(
                candidate_x - active_x,
                candidate_y - active_y,
            )

            if distance_to_active_goal > self.active_goal_match_radius_m:
                continue

            path_length = self.finite_path_length(candidate)

            try:
                candidate_region_id = int(candidate.get("region_id", -999999))
            except (TypeError, ValueError):
                candidate_region_id = -999999

            item = (distance_to_active_goal, path_length, candidate)

            any_region_matches.append(item)

            if candidate_region_id == active_region_id:
                same_region_matches.append(item)

        if same_region_matches:
            same_region_matches.sort(key=lambda x: (x[0], x[1]))
            return same_region_matches[0][2]

        if any_region_matches:
            any_region_matches.sort(key=lambda x: (x[0], x[1]))
            return any_region_matches[0][2]

        return None

    def refresh_active_goal_from_latest_candidates(self) -> bool:
        """
        Refresh the active goal/path so the robot follows the live frontier
        instead of a stale coordinate.

        Returns True if a live matching candidate was found.
        """
        match = self.find_current_match_for_active_candidate()

        if match is None:
            return False

        self.active_candidate = match
        self.active_goal = self.candidate_to_pose_stamped(
            self.latest_candidates_payload,
            match,
        )
        self.active_path = self.candidate_to_path(
            self.latest_candidates_payload,
            match,
        )

        return True

    def active_candidate_invalid_for_too_long(self) -> bool:
        """
        End the current RL action only if no corresponding live frontier can be
        matched for longer than goal_invalid_grace_s.

        If a matching live frontier exists, refresh the active goal/path to
        that current candidate.
        """
        if (
            self.active_goal is None
            or self.active_candidate is None
            or self.latest_candidates_payload is None
        ):
            return False

        still_valid = self.refresh_active_goal_from_latest_candidates()
        now_s = self.now_seconds()

        if still_valid:
            self.active_candidate_invalid_since_s = None
            return False

        if self.active_candidate_invalid_since_s is None:
            self.active_candidate_invalid_since_s = now_s
            return False

        return (now_s - self.active_candidate_invalid_since_s) >= self.goal_invalid_grace_s

    def compute_known_area_m2(self, grid: OccupancyGrid) -> float:
        data = np.asarray(grid.data, dtype=np.int16)
        known_cells = int(np.count_nonzero(data >= 0))
        return known_cells * grid.info.resolution * grid.info.resolution

    def lookup_robot_xy(self) -> Optional[Tuple[float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except TransformException:
            return None

        return (
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
        )

    def finite_path_length(self, candidate: Dict[str, Any]) -> float:
        value = candidate.get("path_length_m", None)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("inf")

        if not math.isfinite(value):
            return float("inf")

        return value

    def prune_failed_goal_cooldowns(self) -> None:
        now_s = self.now_seconds()
        self.failed_goal_cooldowns = [
            item for item in self.failed_goal_cooldowns
            if float(item.get("expires_at_s", 0.0)) > now_s
        ]

    def candidate_in_failed_goal_cooldown(self, candidate: Dict[str, Any]) -> bool:
        self.prune_failed_goal_cooldowns()

        try:
            candidate_x = float(candidate.get("goal_x"))
            candidate_y = float(candidate.get("goal_y"))
        except (TypeError, ValueError):
            return False

        for item in self.failed_goal_cooldowns:
            dx = candidate_x - float(item["x"])
            dy = candidate_y - float(item["y"])

            if math.hypot(dx, dy) <= self.failed_goal_radius_m:
                return True

        return False

    def add_failed_goal_cooldown(self, end_reason: str) -> None:
        if self.active_goal is None and self.active_candidate is None:
            return

        if self.active_goal is not None:
            x = float(self.active_goal.pose.position.x)
            y = float(self.active_goal.pose.position.y)
        else:
            try:
                x = float(self.active_candidate.get("goal_x"))
                y = float(self.active_candidate.get("goal_y"))
            except (TypeError, ValueError, AttributeError):
                return

        now_s = self.now_seconds()
        expires_at_s = now_s + self.failed_goal_cooldown_s

        # Merge with an existing cooldown if it is nearby.
        for item in self.failed_goal_cooldowns:
            if math.hypot(x - float(item["x"]), y - float(item["y"])) <= self.failed_goal_radius_m:
                item["x"] = x
                item["y"] = y
                item["expires_at_s"] = expires_at_s
                item["last_reason"] = end_reason
                item["count"] = int(item.get("count", 1)) + 1
                return

        self.failed_goal_cooldowns.append(
            {
                "x": x,
                "y": y,
                "expires_at_s": expires_at_s,
                "last_reason": end_reason,
                "count": 1,
            }
        )

    def prune_failed_goal_cooldowns(self) -> None:
        now_s = self.now_seconds()

        self.failed_goal_cooldowns = [
            item
            for item in self.failed_goal_cooldowns
            if float(item.get("expires_at_s", 0.0)) > now_s
        ]

    def candidate_in_failed_goal_cooldown(self, candidate: Dict[str, Any]) -> bool:
        self.prune_failed_goal_cooldowns()

        try:
            candidate_x = float(candidate.get("goal_x"))
            candidate_y = float(candidate.get("goal_y"))
        except (TypeError, ValueError):
            return False

        for item in self.failed_goal_cooldowns:
            dx = candidate_x - float(item["x"])
            dy = candidate_y - float(item["y"])

            if math.hypot(dx, dy) <= self.failed_goal_radius_m:
                return True

        return False

    def add_failed_goal_cooldown(self, end_reason: str) -> None:
        if self.active_goal is None and self.active_candidate is None:
            return

        if self.active_goal is not None:
            x = float(self.active_goal.pose.position.x)
            y = float(self.active_goal.pose.position.y)
        else:
            try:
                x = float(self.active_candidate.get("goal_x"))
                y = float(self.active_candidate.get("goal_y"))
            except (TypeError, ValueError, AttributeError):
                return

        now_s = self.now_seconds()
        expires_at_s = now_s + self.failed_goal_cooldown_s

        for item in self.failed_goal_cooldowns:
            if math.hypot(x - float(item["x"]), y - float(item["y"])) <= self.failed_goal_radius_m:
                item["x"] = x
                item["y"] = y
                item["expires_at_s"] = expires_at_s
                item["last_reason"] = end_reason
                item["count"] = int(item.get("count", 1)) + 1
                return

        self.failed_goal_cooldowns.append(
            {
                "x": x,
                "y": y,
                "expires_at_s": expires_at_s,
                "last_reason": end_reason,
                "count": 1,
            }
        )

    def preselect_rl_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.prune_failed_goal_cooldowns()

        valid_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("goal_x") is not None
            and candidate.get("goal_y") is not None
            and math.isfinite(self.finite_path_length(candidate))
            and not self.candidate_in_failed_goal_cooldown(candidate)
        ]

        if not valid_candidates:
            return []

        regions: Dict[int, List[Dict[str, Any]]] = {}

        for candidate in valid_candidates:
            try:
                region_id = int(candidate.get("region_id", -1))
            except (TypeError, ValueError):
                region_id = -1

            regions.setdefault(region_id, []).append(candidate)

        for region_candidates in regions.values():
            region_candidates.sort(
                key=lambda candidate: (
                    self.finite_path_length(candidate),
                    int(candidate.get("candidate_index", 10**9)),
                )
            )

        region_ids = sorted(
            regions.keys(),
            key=lambda region_id: (
                self.finite_path_length(regions[region_id][0]),
                region_id,
            ),
        )

        selected: List[Dict[str, Any]] = []
        depth = 0

        while len(selected) < self.max_rl_candidates:
            added_this_round = False

            for region_id in region_ids:
                region_candidates = regions[region_id]

                if depth >= len(region_candidates):
                    continue

                selected.append(region_candidates[depth])
                added_this_round = True

                if len(selected) >= self.max_rl_candidates:
                    break

            if not added_this_round:
                break

            depth += 1

        return selected

    def build_rl_observation(self, candidates: List[Dict[str, Any]]) -> List[List[float]]:
        observation: List[List[float]] = []

        for candidate in candidates[: self.max_rl_candidates]:
            row: List[float] = []

            for key in self.RL_FEATURE_KEYS:
                value = candidate.get(key, 0.0)

                if value is None:
                    value = 0.0

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = 0.0

                if not math.isfinite(value):
                    value = 0.0

                row.append(value)

            observation.append(row)

        while len(observation) < self.max_rl_candidates:
            observation.append([0.0] * len(self.RL_FEATURE_KEYS))

        return observation

    def choose_nearest_action(self, candidates: List[Dict[str, Any]]) -> Optional[int]:
        best_index = None
        best_path = float("inf")

        for i, candidate in enumerate(candidates):
            path_length = self.finite_path_length(candidate)
            if path_length < best_path:
                best_path = path_length
                best_index = i

        return best_index

    def candidate_to_pose_stamped(
        self,
        payload: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(payload.get("map_frame", self.map_frame))

        stamp_sec = int(payload.get("stamp_sec", 0))
        stamp_nanosec = int(payload.get("stamp_nanosec", 0))
        pose.header.stamp.sec = stamp_sec
        pose.header.stamp.nanosec = stamp_nanosec

        goal_x = float(candidate["goal_x"])
        goal_y = float(candidate["goal_y"])
        goal_yaw = float(candidate.get("goal_yaw", 0.0))

        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.25
        pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        pose.pose.orientation.w = math.cos(goal_yaw / 2.0)

        return pose

    def candidate_to_path(
        self,
        payload: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> NavPath:
        path = NavPath()
        path.header.frame_id = str(payload.get("map_frame", self.map_frame))

        stamp_sec = int(payload.get("stamp_sec", 0))
        stamp_nanosec = int(payload.get("stamp_nanosec", 0))
        path.header.stamp.sec = stamp_sec
        path.header.stamp.nanosec = stamp_nanosec

        path_poses = candidate.get("path_poses", [])
        if not isinstance(path_poses, list):
            path_poses = []

        for item in path_poses:
            if not isinstance(item, dict):
                continue

            try:
                x = float(item["x"])
                y = float(item["y"])
                z = float(item.get("z", 0.05))
            except (KeyError, TypeError, ValueError):
                continue

            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def write_observation_jsonl(
        self,
        observation: List[List[float]],
        valid_action_mask: List[int],
        preselected: List[Dict[str, Any]],
        action_index: int,
    ) -> None:
        record = {
            "transition_index": self.transition_count,
            "time_s": self.now_seconds(),
            "feature_keys": self.RL_FEATURE_KEYS,
            "observation": observation,
            "valid_action_mask": valid_action_mask,
            "preselected_candidate_indices": [
                int(candidate.get("candidate_index", -1))
                for candidate in preselected
            ],
            "chosen_action_index": action_index,
        }

        self.observations_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.observations_file.flush()

    def log_invalid_action(
        self,
        observation: List[List[float]],
        valid_action_mask: List[int],
        action_index: int,
    ) -> None:
        now_s = self.now_seconds()

        self.transitions_writer.writerow(
            {
                "transition_index": self.transition_count,
                "start_time_s": now_s,
                "end_time_s": now_s,
                "duration_s": 0.0,
                "action_index": action_index,
                "candidate_index": "",
                "cluster_id": "",
                "region_id": "",
                "goal_x": "",
                "goal_y": "",
                "path_length_m": "",
                "known_area_before_m2": "",
                "known_area_after_m2": "",
                "known_area_delta_m2": 0.0,
                "distance_travelled_m": 0.0,
                "distance_penalty_lambda": self.distance_penalty_lambda,
                "failure_penalty": self.invalid_action_penalty,
                "reward": -self.invalid_action_penalty,
                "end_reason": "invalid_action",
                "preselected_candidate_indices": "",
                "valid_action_mask": ";".join(str(x) for x in valid_action_mask),
                "failed_goal_cooldown_count": len(self.failed_goal_cooldowns),
            }
        )
        self.transitions_file.flush()
        self.transition_count += 1

    def publish_status(self) -> None:
        status = {
            "active_action": self.active_action,
            "transition_count": self.transition_count,
            "active_action_index": self.active_action_index,
            "active_candidate_index": (
                self.active_candidate.get("candidate_index")
                if self.active_candidate is not None
                else None
            ),
            "action_distance_m": self.action_distance_m,
            "max_action_duration_s": self.max_action_duration_s,
            "active_action_timeout_s": self.active_action_timeout_s,
            "min_action_timeout_s": self.min_action_timeout_s,
            "expected_navigation_speed_mps": self.expected_navigation_speed_mps,
            "action_timeout_scale": self.action_timeout_scale,
            "min_action_duration_s": self.min_action_duration_s,
            "min_goal_distance_reduction_m": self.min_goal_distance_reduction_m,
            "min_known_area_gain_m2": self.min_known_area_gain_m2,
            "min_robot_motion_m": self.min_robot_motion_m,
            "failed_goal_cooldown_s": self.failed_goal_cooldown_s,
            "failed_goal_radius_m": self.failed_goal_radius_m,
            "failed_goal_cooldown_count": len(self.failed_goal_cooldowns),
            "goal_invalid_grace_s": self.goal_invalid_grace_s,
            "active_goal_match_radius_m": self.active_goal_match_radius_m,
            "active_candidate_invalid_since_s": self.active_candidate_invalid_since_s,
            "reward_formula": "known_area_delta_m2 - lambda_distance * distance_m - failure_penalty",
            "distance_penalty_lambda": self.distance_penalty_lambda,
            "log_dir": str(self.run_dir),
            "latest_candidates_available": self.latest_candidates_payload is not None,
            "latest_grid_available": self.latest_grid is not None,
            "supervisor_state": self.latest_supervisor_status.get("state", ""),
        }

        out = String()
        out.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(out)

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def destroy_node(self):
        try:
            self.transitions_file.flush()
            self.transitions_file.close()
            self.observations_file.flush()
            self.observations_file.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RLTrainingEnvStub()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
