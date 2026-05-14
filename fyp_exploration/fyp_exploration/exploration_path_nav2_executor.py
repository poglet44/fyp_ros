#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from std_msgs.msg import String

from tf2_ros import Buffer, TransformException, TransformListener


@dataclass
class BlacklistedGoal:
    x: float
    y: float
    stamp: Time


class ExplorationPathNav2Executor(Node):
    """
    Path-based Nav2 executor for frontier exploration.

    Instead of sending the final /selected_frontier_goal directly to Nav2, this node:
      1. subscribes to /frontier_path
      2. finds the robot's closest point on that path
      3. selects a short lookahead pose along the path
      4. sends that short subgoal to Nav2

    This keeps Nav2 from trying to solve the whole room/doorway route from one far-away
    frontier goal.
    """

    def __init__(self) -> None:
        super().__init__("exploration_path_nav2_executor")

        self.declare_parameter("frontier_path_topic", "/frontier_path")
        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("exploration_status_topic", "/exploration_status")
        self.declare_parameter("navigate_action_name", "/navigate_to_pose")

        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("path_stale_timeout_s", 3.0)

        self.declare_parameter("path_lookahead_distance_m", 0.90)
        self.declare_parameter("subgoal_reached_distance_m", 0.35)
        self.declare_parameter("subgoal_replace_distance_m", 0.75)

        self.declare_parameter("goal_timeout_s", 45.0)
        self.declare_parameter("control_period_s", 0.20)
        self.declare_parameter("wait_for_server_timeout_s", 5.0)

        self.declare_parameter("use_selected_goal_fallback", True)

        self.declare_parameter("abort_blacklist_radius_m", 0.60)
        self.declare_parameter("abort_blacklist_timeout_s", 30.0)

        self.declare_parameter(
            "terminal_status_values",
            ["COMPLETE", "NO_REACHABLE_FRONTIERS", "EXPLORATION_COMPLETE"],
        )

        self.frontier_path_topic = self.get_parameter("frontier_path_topic").value
        self.selected_goal_topic = self.get_parameter("selected_goal_topic").value
        self.exploration_status_topic = self.get_parameter("exploration_status_topic").value
        self.navigate_action_name = self.get_parameter("navigate_action_name").value

        self.robot_frame = self.get_parameter("robot_frame").value
        self.path_stale_timeout_s = float(self.get_parameter("path_stale_timeout_s").value)

        self.path_lookahead_distance_m = float(self.get_parameter("path_lookahead_distance_m").value)
        self.subgoal_reached_distance_m = float(self.get_parameter("subgoal_reached_distance_m").value)
        self.subgoal_replace_distance_m = float(self.get_parameter("subgoal_replace_distance_m").value)

        self.goal_timeout_s = float(self.get_parameter("goal_timeout_s").value)
        self.control_period_s = float(self.get_parameter("control_period_s").value)
        self.wait_for_server_timeout_s = float(self.get_parameter("wait_for_server_timeout_s").value)

        self.use_selected_goal_fallback = bool(self.get_parameter("use_selected_goal_fallback").value)

        self.abort_blacklist_radius_m = float(self.get_parameter("abort_blacklist_radius_m").value)
        self.abort_blacklist_timeout_s = float(self.get_parameter("abort_blacklist_timeout_s").value)

        self.terminal_status_values = set(self.get_parameter("terminal_status_values").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, self.navigate_action_name)

        self.latest_path: Optional[Path] = None
        self.latest_path_time: Optional[Time] = None

        self.latest_selected_goal: Optional[PoseStamped] = None
        self.latest_selected_goal_time: Optional[Time] = None

        self.latest_status: Optional[str] = None
        self.stop_requested = False

        self.active_goal_handle = None
        self.active_goal_pose: Optional[PoseStamped] = None
        self.active_goal_sent_time: Optional[Time] = None
        self.cancel_in_progress = False

        self.blacklisted_goals: List[BlacklistedGoal] = []

        self.create_subscription(Path, self.frontier_path_topic, self.path_callback, 10)
        self.create_subscription(PoseStamped, self.selected_goal_topic, self.selected_goal_callback, 10)
        self.create_subscription(String, self.exploration_status_topic, self.status_callback, 10)

        self.create_timer(self.control_period_s, self.control_loop)

        self.get_logger().info(
            "Path-based Nav2 executor started:\n"
            f"  frontier_path_topic={self.frontier_path_topic}\n"
            f"  selected_goal_topic={self.selected_goal_topic}\n"
            f"  exploration_status_topic={self.exploration_status_topic}\n"
            f"  navigate_action_name={self.navigate_action_name}\n"
            f"  robot_frame={self.robot_frame}\n"
            f"  path_lookahead_distance_m={self.path_lookahead_distance_m:.2f}\n"
            f"  subgoal_reached_distance_m={self.subgoal_reached_distance_m:.2f}\n"
            f"  abort_blacklist_radius_m={self.abort_blacklist_radius_m:.2f}\n"
            f"  abort_blacklist_timeout_s={self.abort_blacklist_timeout_s:.2f}"
        )

    def path_callback(self, msg: Path) -> None:
        if not msg.poses:
            return

        if not msg.header.frame_id:
            self.get_logger().warn("Received /frontier_path with empty frame_id. Ignoring.")
            return

        self.latest_path = msg
        self.latest_path_time = self.get_clock().now()

    def selected_goal_callback(self, msg: PoseStamped) -> None:
        if not msg.header.frame_id:
            return

        self.latest_selected_goal = msg
        self.latest_selected_goal_time = self.get_clock().now()

    def status_callback(self, msg: String) -> None:
        self.latest_status = msg.data

        if msg.data in self.terminal_status_values:
            if not self.stop_requested:
                self.get_logger().warn(
                    f"Received terminal exploration status '{msg.data}'. Canceling active Nav2 goal."
                )
            self.stop_requested = True
            self.cancel_active_goal()
        else:
            self.stop_requested = False

    def control_loop(self) -> None:
        now = self.get_clock().now()

        self.cleanup_blacklist(now)

        if self.stop_requested:
            return

        if not self.nav_client.server_is_ready():
            available = self.nav_client.wait_for_server(
                timeout_sec=self.wait_for_server_timeout_s
            )
            if not available:
                self.get_logger().warn("Nav2 NavigateToPose action server is not available.")
                return

        if self.active_goal_handle is not None:
            self.monitor_active_goal(now)
            return

        subgoal = self.compute_next_subgoal(now)
        if subgoal is None:
            return

        self.send_goal(subgoal)

    def monitor_active_goal(self, now: Time) -> None:
        if self.active_goal_pose is None or self.active_goal_sent_time is None:
            return

        if now - self.active_goal_sent_time > Duration(seconds=self.goal_timeout_s):
            self.get_logger().warn("Active Nav2 subgoal timed out. Canceling and blacklisting.")
            self.blacklist_pose(self.active_goal_pose)
            self.cancel_active_goal()
            return

        robot_xy = self.get_robot_xy(self.active_goal_pose.header.frame_id)
        if robot_xy is None:
            return

        dx = robot_xy[0] - self.active_goal_pose.pose.position.x
        dy = robot_xy[1] - self.active_goal_pose.pose.position.y
        dist = math.hypot(dx, dy)

        if dist <= self.subgoal_reached_distance_m:
            self.get_logger().info(
                f"Subgoal reached by distance check ({dist:.2f} m). Canceling current subgoal and advancing."
            )
            self.cancel_active_goal()

    def compute_next_subgoal(self, now: Time) -> Optional[PoseStamped]:
        path_goal = self.compute_path_lookahead_subgoal(now)
        if path_goal is not None:
            return path_goal

        if not self.use_selected_goal_fallback:
            return None

        return self.compute_selected_goal_fallback(now)

    def compute_path_lookahead_subgoal(self, now: Time) -> Optional[PoseStamped]:
        if self.latest_path is None or self.latest_path_time is None:
            return None

        if now - self.latest_path_time > Duration(seconds=self.path_stale_timeout_s):
            return None

        path = self.latest_path
        if not path.poses:
            return None

        frame_id = path.header.frame_id
        robot_xy = self.get_robot_xy(frame_id)
        if robot_xy is None:
            return None

        closest_i = self.find_closest_path_index(path, robot_xy)
        target_i = self.find_lookahead_index(path, closest_i, self.path_lookahead_distance_m)

        # If the first lookahead target is blacklisted, walk further along the path.
        for i in range(target_i, len(path.poses)):
            candidate = PoseStamped()
            candidate.header.frame_id = frame_id
            candidate.header.stamp = now.to_msg()
            candidate.pose = path.poses[i].pose

            if not self.is_blacklisted(candidate):
                return candidate

        return None

    def compute_selected_goal_fallback(self, now: Time) -> Optional[PoseStamped]:
        if self.latest_selected_goal is None or self.latest_selected_goal_time is None:
            return None

        if now - self.latest_selected_goal_time > Duration(seconds=self.path_stale_timeout_s):
            return None

        if self.is_blacklisted(self.latest_selected_goal):
            return None

        goal = PoseStamped()
        goal.header.frame_id = self.latest_selected_goal.header.frame_id
        goal.header.stamp = now.to_msg()
        goal.pose = self.latest_selected_goal.pose
        return goal

    def find_closest_path_index(self, path: Path, robot_xy: Tuple[float, float]) -> int:
        best_i = 0
        best_d = float("inf")

        rx, ry = robot_xy

        for i, pose_stamped in enumerate(path.poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            d = math.hypot(px - rx, py - ry)
            if d < best_d:
                best_d = d
                best_i = i

        return best_i

    def find_lookahead_index(self, path: Path, start_i: int, lookahead_m: float) -> int:
        if start_i >= len(path.poses) - 1:
            return start_i

        accumulated = 0.0

        for i in range(start_i, len(path.poses) - 1):
            p0 = path.poses[i].pose.position
            p1 = path.poses[i + 1].pose.position
            accumulated += math.hypot(p1.x - p0.x, p1.y - p0.y)

            if accumulated >= lookahead_m:
                return i + 1

        return len(path.poses) - 1

    def get_robot_xy(self, target_frame: str) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                self.robot_frame,
                Time(),
                timeout=Duration(seconds=0.15),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"Could not transform {self.robot_frame} into {target_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        return (
            tf.transform.translation.x,
            tf.transform.translation.y,
        )

    def send_goal(self, goal_pose: PoseStamped) -> None:
        if self.active_goal_handle is not None:
            return

        if self.is_blacklisted(goal_pose):
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            "Sending Nav2 path subgoal: "
            f"x={goal_pose.pose.position.x:.2f}, "
            f"y={goal_pose.pose.position.y:.2f}, "
            f"frame={goal_pose.header.frame_id}"
        )

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

        self.active_goal_pose = goal_pose
        self.active_goal_sent_time = self.get_clock().now()

    def goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"NavigateToPose goal request failed: {exc}")
            self.active_goal_handle = None
            self.active_goal_pose = None
            self.active_goal_sent_time = None
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 rejected path subgoal.")
            if self.active_goal_pose is not None:
                self.blacklist_pose(self.active_goal_pose)
            self.active_goal_handle = None
            self.active_goal_pose = None
            self.active_goal_sent_time = None
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future) -> None:
        status = None

        try:
            result = future.result()
            status = result.status
        except Exception as exc:
            self.get_logger().error(f"Failed to get Nav2 result: {exc}")

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Nav2 path subgoal succeeded.")
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn("Nav2 path subgoal aborted. Blacklisting this subgoal area.")
            if self.active_goal_pose is not None:
                self.blacklist_pose(self.active_goal_pose)
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("Nav2 path subgoal canceled.")
        elif status is not None:
            self.get_logger().warn(f"Nav2 path subgoal finished with status code {status}.")

        self.active_goal_handle = None
        self.active_goal_pose = None
        self.active_goal_sent_time = None
        self.cancel_in_progress = False

    def cancel_active_goal(self) -> None:
        if self.active_goal_handle is None:
            self.active_goal_pose = None
            self.active_goal_sent_time = None
            self.cancel_in_progress = False
            return

        if self.cancel_in_progress:
            return

        self.cancel_in_progress = True
        cancel_future = self.active_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_callback)

    def cancel_callback(self, future) -> None:
        try:
            _ = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Failed to cancel Nav2 goal cleanly: {exc}")

        self.active_goal_handle = None
        self.active_goal_pose = None
        self.active_goal_sent_time = None
        self.cancel_in_progress = False

    def blacklist_pose(self, pose: PoseStamped) -> None:
        self.blacklisted_goals.append(
            BlacklistedGoal(
                x=pose.pose.position.x,
                y=pose.pose.position.y,
                stamp=self.get_clock().now(),
            )
        )

    def cleanup_blacklist(self, now: Time) -> None:
        timeout = Duration(seconds=self.abort_blacklist_timeout_s)
        self.blacklisted_goals = [
            item for item in self.blacklisted_goals
            if now - item.stamp <= timeout
        ]

    def is_blacklisted(self, pose: PoseStamped) -> bool:
        px = pose.pose.position.x
        py = pose.pose.position.y

        for item in self.blacklisted_goals:
            if math.hypot(px - item.x, py - item.y) <= self.abort_blacklist_radius_m:
                return True

        return False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationPathNav2Executor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
