#!/usr/bin/env python3

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


STATE_EXPLORING = "EXPLORING"
STATE_NO_CANDIDATES_PENDING = "NO_CANDIDATES_PENDING"
STATE_RECOVERY_LOCAL_RECHECK = "RECOVERY_LOCAL_RECHECK"
STATE_RECOVERY_CHECKPOINT = "RECOVERY_CHECKPOINT"
STATE_RECOVERY_RETURN_START = "RECOVERY_RETURN_START"
STATE_NO_REACHABLE_CANDIDATES_AFTER_RECOVERY = "NO_REACHABLE_CANDIDATES_AFTER_RECOVERY"


@dataclass
class RobotPose2D:
    x: float
    y: float
    yaw: float


@dataclass
class RecoveryCheckpoint:
    checkpoint_id: int
    pose: PoseStamped
    map_count: int
    ros_time_ns: int
    num_candidate_goals: int
    raw_frontier_cells: int
    filtered_frontier_cells: int
    travel_distance_m: float
    distance_from_start_m: float


class ExplorationSupervisor(Node):
    """
    ROS2 port of the ROS1 exploration supervisor.

    Responsibility:
      - Watch /frontier_detector/status
      - Detect no-candidate frontier starvation
      - Store candidate-rich checkpoints
      - Publish recovery state/status
      - Optionally publish recovery target goals

    Not responsible for:
      - Frontier detection
      - Final goal arbitration
      - Nav2 command execution
      - Low-level velocity control
    """

    def __init__(self):
        super().__init__("exploration_supervisor")

        # Topics
        self.declare_parameter("detector_status_topic", "/frontier_detector/status")
        self.declare_parameter("supervisor_status_topic", "/exploration_supervisor/status")
        self.declare_parameter("recovery_state_topic", "/exploration_supervisor/recovery_state")
        self.declare_parameter("recovery_target_goal_topic", "/exploration_supervisor/recovery_target_goal")
        self.declare_parameter("executor_status_topic", "/spot_frontier_executor/status")

        # Frames
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("tf_lookup_timeout_s", 0.20)

        # Recovery behaviour
        self.declare_parameter("no_candidates_grace_cycles", 3)
        self.declare_parameter("recovery_mode", "return_to_checkpoint")
        self.declare_parameter("publish_recovery_target", True)
        self.declare_parameter("local_recheck_cycles", 5)

        # Checkpoints
        self.declare_parameter("checkpoint_min_candidates", 2)
        self.declare_parameter("checkpoint_min_spacing_m", 1.5)
        self.declare_parameter("max_checkpoints", 20)
        self.declare_parameter("recovery_progress_fractions", [0.75, 0.50, 0.25])
        self.declare_parameter("checkpoint_min_return_distance_m", 1.0)

        # Executor interpretation
        self.declare_parameter("executor_goal_reached_reason", "goal_reached")
        self.declare_parameter("recovery_target_switch_grace_s", 3.0)
        self.declare_parameter("active_target_match_tolerance_m", 0.75)
        self.declare_parameter("active_target_goal_reached_tolerance_m", 0.75)

        # Travel integration
        self.declare_parameter("min_progress_update_distance_m", 0.03)

        self.detector_status_topic = self.get_parameter("detector_status_topic").value
        self.supervisor_status_topic = self.get_parameter("supervisor_status_topic").value
        self.recovery_state_topic = self.get_parameter("recovery_state_topic").value
        self.recovery_target_goal_topic = self.get_parameter("recovery_target_goal_topic").value
        self.executor_status_topic = self.get_parameter("executor_status_topic").value

        self.map_frame = self.get_parameter("map_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)

        self.no_candidates_grace_cycles = int(self.get_parameter("no_candidates_grace_cycles").value)
        self.recovery_mode = str(self.get_parameter("recovery_mode").value)
        self.publish_recovery_target_enabled = bool(self.get_parameter("publish_recovery_target").value)
        self.local_recheck_cycles = int(self.get_parameter("local_recheck_cycles").value)

        self.checkpoint_min_candidates = int(self.get_parameter("checkpoint_min_candidates").value)
        self.checkpoint_min_spacing_m = float(self.get_parameter("checkpoint_min_spacing_m").value)
        self.max_checkpoints = int(self.get_parameter("max_checkpoints").value)
        self.recovery_progress_fractions = [
            float(v) for v in self.get_parameter("recovery_progress_fractions").value
        ]
        self.checkpoint_min_return_distance_m = float(
            self.get_parameter("checkpoint_min_return_distance_m").value
        )

        self.executor_goal_reached_reason = str(
            self.get_parameter("executor_goal_reached_reason").value
        )
        self.recovery_target_switch_grace_s = float(
            self.get_parameter("recovery_target_switch_grace_s").value
        )
        self.active_target_match_tolerance_m = float(
            self.get_parameter("active_target_match_tolerance_m").value
        )
        self.active_target_goal_reached_tolerance_m = float(
            self.get_parameter("active_target_goal_reached_tolerance_m").value
        )

        self.min_progress_update_distance_m = float(
            self.get_parameter("min_progress_update_distance_m").value
        )

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.status_pub = self.create_publisher(String, self.supervisor_status_topic, 10)
        self.recovery_state_pub = self.create_publisher(String, self.recovery_state_topic, 10)
        self.recovery_target_pub = self.create_publisher(PoseStamped, self.recovery_target_goal_topic, 10)

        self.create_subscription(String, self.detector_status_topic, self.detector_status_callback, 10)
        self.create_subscription(String, self.executor_status_topic, self.executor_status_callback, 10)

        self.state = STATE_EXPLORING
        self.no_candidates_count = 0
        self.local_recheck_count = 0

        self.latest_detector_status: Dict = {}
        self.latest_executor_status: Dict = {}

        self.start_pose: Optional[PoseStamped] = None
        self.previous_robot_pose: Optional[RobotPose2D] = None
        self.current_robot_pose: Optional[RobotPose2D] = None
        self.total_travel_distance_m = 0.0

        self.checkpoints: List[RecoveryCheckpoint] = []
        self.next_checkpoint_id = 1

        self.recovery_episode_id = 0
        self.recovery_start_travel_distance_m = 0.0
        self.tried_checkpoint_ids = set()
        self.tried_progress_fractions = set()

        self.active_recovery_target: Optional[PoseStamped] = None
        self.active_recovery_target_type = ""
        self.active_recovery_checkpoint_id: Optional[int] = None
        self.active_recovery_fraction: Optional[float] = None
        self.active_recovery_target_set_time_s: Optional[float] = None

        self.get_logger().info(
            "exploration_supervisor started:\n"
            f"  detector_status_topic={self.detector_status_topic}\n"
            f"  executor_status_topic={self.executor_status_topic}\n"
            f"  recovery_mode={self.recovery_mode}\n"
            f"  publish_recovery_target={self.publish_recovery_target_enabled}\n"
            f"  robot_frame={self.robot_frame}"
        )

    def detector_status_callback(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Ignoring invalid detector status JSON.")
            return

        self.latest_detector_status = status

        pose = self.lookup_robot_pose()
        if pose is not None:
            self.update_travel_distance(pose)

        if self.start_pose is None and self.current_robot_pose is not None:
            self.start_pose = self.robot_pose_to_pose_stamped(self.current_robot_pose)

        self.maybe_store_checkpoint(status)

        num_candidate_goals = int(status.get("num_candidate_goals", 0))

        if num_candidate_goals > 0:
            if self.state != STATE_EXPLORING:
                self.get_logger().info("Candidates available again. Returning to EXPLORING.")
            self.reset_recovery()
            self.state = STATE_EXPLORING
            self.no_candidates_count = 0
            self.publish_all_status()
            return

        self.no_candidates_count += 1

        if self.recovery_mode in ("none", "diagnostic_only"):
            if self.no_candidates_count >= self.no_candidates_grace_cycles:
                self.state = STATE_NO_CANDIDATES_PENDING
            self.publish_all_status()
            return

        if self.state == STATE_EXPLORING:
            if self.no_candidates_count < self.no_candidates_grace_cycles:
                self.state = STATE_NO_CANDIDATES_PENDING
            else:
                self.start_recovery_episode()
                self.state = STATE_RECOVERY_LOCAL_RECHECK

        elif self.state == STATE_NO_CANDIDATES_PENDING:
            if self.no_candidates_count >= self.no_candidates_grace_cycles:
                self.start_recovery_episode()
                self.state = STATE_RECOVERY_LOCAL_RECHECK

        elif self.state == STATE_RECOVERY_LOCAL_RECHECK:
            self.local_recheck_count += 1
            if self.local_recheck_count >= self.local_recheck_cycles:
                self.choose_next_recovery_target()

        elif self.state == STATE_RECOVERY_CHECKPOINT:
            if self.executor_reports_active_target_reached():
                self.choose_next_recovery_target()

        elif self.state == STATE_RECOVERY_RETURN_START:
            if self.executor_reports_active_target_reached():
                self.state = STATE_NO_REACHABLE_CANDIDATES_AFTER_RECOVERY

        self.publish_all_status()

    def executor_status_callback(self, msg: String) -> None:
        try:
            self.latest_executor_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Ignoring invalid executor status JSON.")

    def start_recovery_episode(self) -> None:
        self.recovery_episode_id += 1
        self.local_recheck_count = 0
        self.recovery_start_travel_distance_m = self.total_travel_distance_m
        self.tried_checkpoint_ids.clear()
        self.tried_progress_fractions.clear()
        self.clear_active_recovery_target()
        self.get_logger().warn(
            f"Starting recovery episode {self.recovery_episode_id}: no candidate goals."
        )

    def reset_recovery(self) -> None:
        self.local_recheck_count = 0
        self.clear_active_recovery_target()

    def clear_active_recovery_target(self) -> None:
        self.active_recovery_target = None
        self.active_recovery_target_type = ""
        self.active_recovery_checkpoint_id = None
        self.active_recovery_fraction = None
        self.active_recovery_target_set_time_s = None

    def choose_next_recovery_target(self) -> None:
        checkpoint, fraction = self.select_recovery_checkpoint()

        if checkpoint is not None:
            self.set_recovery_target(
                pose=checkpoint.pose,
                target_type="checkpoint",
                checkpoint_id=checkpoint.checkpoint_id,
                fraction=fraction,
            )
            self.tried_checkpoint_ids.add(checkpoint.checkpoint_id)
            if fraction is not None:
                self.tried_progress_fractions.add(fraction)
            self.state = STATE_RECOVERY_CHECKPOINT
            return

        if self.start_pose is not None:
            self.set_recovery_target(
                pose=self.start_pose,
                target_type="return_start",
                checkpoint_id=None,
                fraction=None,
            )
            self.state = STATE_RECOVERY_RETURN_START
            return

        self.state = STATE_NO_REACHABLE_CANDIDATES_AFTER_RECOVERY

    def set_recovery_target(
        self,
        pose: PoseStamped,
        target_type: str,
        checkpoint_id: Optional[int],
        fraction: Optional[float],
    ) -> None:
        target = PoseStamped()
        target.header.frame_id = self.map_frame
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose = pose.pose

        self.active_recovery_target = target
        self.active_recovery_target_type = target_type
        self.active_recovery_checkpoint_id = checkpoint_id
        self.active_recovery_fraction = fraction
        self.active_recovery_target_set_time_s = self.now_seconds()

        self.get_logger().warn(
            f"Recovery target selected: type={target_type}, "
            f"checkpoint_id={checkpoint_id}, fraction={fraction}, "
            f"x={target.pose.position.x:.2f}, y={target.pose.position.y:.2f}"
        )

    def select_recovery_checkpoint(self) -> Tuple[Optional[RecoveryCheckpoint], Optional[float]]:
        if not self.checkpoints:
            return None, None

        current_pose = self.current_robot_pose
        if current_pose is None:
            return None, None

        for fraction in self.recovery_progress_fractions:
            if fraction in self.tried_progress_fractions:
                continue

            desired_distance = self.recovery_start_travel_distance_m * fraction

            best_checkpoint = None
            best_score = float("inf")

            for checkpoint in self.checkpoints:
                if checkpoint.checkpoint_id in self.tried_checkpoint_ids:
                    continue

                distance_to_current = self.distance(
                    current_pose.x,
                    current_pose.y,
                    checkpoint.pose.pose.position.x,
                    checkpoint.pose.pose.position.y,
                )

                if distance_to_current < self.checkpoint_min_return_distance_m:
                    continue

                progress_error = abs(checkpoint.travel_distance_m - desired_distance)
                candidate_bonus = 0.05 * float(checkpoint.num_candidate_goals)
                score = progress_error - candidate_bonus

                if score < best_score:
                    best_score = score
                    best_checkpoint = checkpoint

            if best_checkpoint is not None:
                return best_checkpoint, fraction

            self.tried_progress_fractions.add(fraction)

        return None, None

    def maybe_store_checkpoint(self, detector_status: Dict) -> None:
        if self.state != STATE_EXPLORING:
            return

        if not bool(detector_status.get("planning_ran", False)):
            return

        if self.current_robot_pose is None:
            return

        num_candidate_goals = int(detector_status.get("num_candidate_goals", 0))
        if num_candidate_goals < self.checkpoint_min_candidates:
            return

        pose_msg = self.robot_pose_to_pose_stamped(self.current_robot_pose)

        if self.checkpoints:
            last_pose = self.checkpoints[-1].pose.pose.position
            spacing = self.distance(
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                last_pose.x,
                last_pose.y,
            )

            if spacing < self.checkpoint_min_spacing_m:
                return

        distance_from_start = 0.0
        if self.start_pose is not None:
            distance_from_start = self.distance(
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                self.start_pose.pose.position.x,
                self.start_pose.pose.position.y,
            )

        checkpoint = RecoveryCheckpoint(
            checkpoint_id=self.next_checkpoint_id,
            pose=pose_msg,
            map_count=int(detector_status.get("map_count", 0)),
            ros_time_ns=self.get_clock().now().nanoseconds,
            num_candidate_goals=num_candidate_goals,
            raw_frontier_cells=int(detector_status.get("raw_frontier_cells", 0)),
            filtered_frontier_cells=int(detector_status.get("filtered_frontier_cells", 0)),
            travel_distance_m=self.total_travel_distance_m,
            distance_from_start_m=distance_from_start,
        )

        self.next_checkpoint_id += 1
        self.checkpoints.append(checkpoint)

        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]

    def executor_reports_active_target_reached(self) -> bool:
        if self.active_recovery_target is None:
            return False

        if not self.latest_executor_status:
            return False

        if self.active_recovery_target_set_time_s is not None:
            age_s = self.now_seconds() - self.active_recovery_target_set_time_s
            if age_s < self.recovery_target_switch_grace_s:
                return False

        reason = self.latest_executor_status.get("reason", "")
        if reason != self.executor_goal_reached_reason:
            return False

        goal_dist = self.latest_executor_status.get("goal_dist", None)
        if isinstance(goal_dist, (int, float)):
            if goal_dist > self.active_target_goal_reached_tolerance_m:
                return False

        executor_goal_x = self.latest_executor_status.get("goal_x", None)
        executor_goal_y = self.latest_executor_status.get("goal_y", None)

        if isinstance(executor_goal_x, (int, float)) and isinstance(executor_goal_y, (int, float)):
            target_x = self.active_recovery_target.pose.position.x
            target_y = self.active_recovery_target.pose.position.y

            if self.distance(executor_goal_x, executor_goal_y, target_x, target_y) > self.active_target_match_tolerance_m:
                return False

        return True

    def lookup_robot_pose(self) -> Optional[RobotPose2D]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except TransformException:
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        yaw = self.yaw_from_quaternion(q.x, q.y, q.z, q.w)
        return RobotPose2D(x=t.x, y=t.y, yaw=yaw)

    def update_travel_distance(self, pose: RobotPose2D) -> None:
        if self.previous_robot_pose is None:
            self.previous_robot_pose = pose
            self.current_robot_pose = pose
            return

        step = self.distance(
            self.previous_robot_pose.x,
            self.previous_robot_pose.y,
            pose.x,
            pose.y,
        )

        if step >= self.min_progress_update_distance_m:
            self.total_travel_distance_m += step
            self.previous_robot_pose = pose

        self.current_robot_pose = pose

    def robot_pose_to_pose_stamped(self, pose: RobotPose2D) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = pose.x
        msg.pose.position.y = pose.y
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(pose.yaw / 2.0)
        msg.pose.orientation.w = math.cos(pose.yaw / 2.0)
        return msg

    def publish_all_status(self) -> None:
        self.publish_supervisor_status()
        self.publish_recovery_state()

        if (
            self.publish_recovery_target_enabled
            and self.active_recovery_target is not None
            and self.state in (STATE_RECOVERY_CHECKPOINT, STATE_RECOVERY_RETURN_START)
        ):
            target = PoseStamped()
            target.header.frame_id = self.map_frame
            target.header.stamp = self.get_clock().now().to_msg()
            target.pose = self.active_recovery_target.pose
            self.recovery_target_pub.publish(target)

    def publish_supervisor_status(self) -> None:
        latest_checkpoint = self.checkpoints[-1] if self.checkpoints else None

        status = {
            "state": self.state,
            "recovery_mode": self.recovery_mode,
            "publish_recovery_target": self.publish_recovery_target_enabled,

            "no_candidates_count": self.no_candidates_count,
            "no_candidates_grace_cycles": self.no_candidates_grace_cycles,
            "local_recheck_count": self.local_recheck_count,
            "local_recheck_cycles": self.local_recheck_cycles,

            "recovery_episode_id": self.recovery_episode_id,
            "recovery_start_travel_distance_m": self.recovery_start_travel_distance_m,

            "total_travel_distance_m": self.total_travel_distance_m,
            "start_pose_available": self.start_pose is not None,
            "robot_pose_available": self.current_robot_pose is not None,
            "robot_x": self.current_robot_pose.x if self.current_robot_pose else None,
            "robot_y": self.current_robot_pose.y if self.current_robot_pose else None,
            "robot_yaw": self.current_robot_pose.yaw if self.current_robot_pose else None,

            "checkpoint_count": len(self.checkpoints),
            "latest_checkpoint_id": latest_checkpoint.checkpoint_id if latest_checkpoint else None,
            "latest_checkpoint_candidates": latest_checkpoint.num_candidate_goals if latest_checkpoint else None,

            "active_recovery_target_available": self.active_recovery_target is not None,
            "active_recovery_target_type": self.active_recovery_target_type,
            "active_recovery_checkpoint_id": self.active_recovery_checkpoint_id,
            "active_recovery_fraction": self.active_recovery_fraction,
            "active_recovery_target_x": (
                self.active_recovery_target.pose.position.x
                if self.active_recovery_target is not None else None
            ),
            "active_recovery_target_y": (
                self.active_recovery_target.pose.position.y
                if self.active_recovery_target is not None else None
            ),

            "detector_map_count": self.latest_detector_status.get("map_count", None),
            "detector_planner_status": self.latest_detector_status.get("planner_status", ""),
            "detector_planning_ran": self.latest_detector_status.get("planning_ran", None),
            "raw_frontier_cells": self.latest_detector_status.get("raw_frontier_cells", None),
            "filtered_frontier_cells": self.latest_detector_status.get("filtered_frontier_cells", None),
            "num_candidate_goals": self.latest_detector_status.get("num_candidate_goals", None),
            "num_frontier_clusters": self.latest_detector_status.get("num_frontier_clusters", None),
            "clusters_total": self.latest_detector_status.get("clusters_total", None),
            "clusters_no_safe_or_reachable_goal": self.latest_detector_status.get("clusters_no_safe_or_reachable_goal", None),
            "clusters_no_path": self.latest_detector_status.get("clusters_no_path", None),
            "clusters_accepted": self.latest_detector_status.get("clusters_accepted", None),

            "executor_state": self.latest_executor_status.get("state", ""),
            "executor_reason": self.latest_executor_status.get("reason", ""),
            "executor_goal_dist": self.latest_executor_status.get("goal_dist", None),

            "map_frame": self.map_frame,
            "robot_frame": self.robot_frame,
        }

        out = String()
        out.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(out)

    def publish_recovery_state(self) -> None:
        state = {
            "state": self.state,
            "recovery_mode": self.recovery_mode,
            "target_available": self.active_recovery_target is not None,
            "target_type": self.active_recovery_target_type,
            "target_checkpoint_id": self.active_recovery_checkpoint_id,
            "target_fraction": self.active_recovery_fraction,
            "target_topic": self.recovery_target_goal_topic,
        }

        out = String()
        out.data = json.dumps(state, sort_keys=True)
        self.recovery_state_pub.publish(out)

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationSupervisor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
