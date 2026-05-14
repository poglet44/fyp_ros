#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String


class ExplorationNav2SimpleExecutor(Node):
    """
    Minimal exploration Nav2 executor.

    Responsibility:
      /selected_frontier_goal -> NavigateToPose

    Rules:
      - send one goal at a time
      - do not spam duplicate goals
      - do not replace active goals
      - after abort, wait briefly before retrying
    """

    def __init__(self) -> None:
        super().__init__("exploration_nav2_simple_executor")

        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("exploration_status_topic", "/exploration_status")
        self.declare_parameter("navigate_action_name", "/navigate_to_pose")

        self.declare_parameter("required_goal_frame", "map")
        self.declare_parameter("enforce_required_goal_frame", True)

        self.declare_parameter("goal_timeout_s", 90.0)
        self.declare_parameter("abort_cooldown_s", 5.0)
        self.declare_parameter("duplicate_goal_tolerance_m", 0.25)

        self.declare_parameter("wait_for_server_timeout_s", 5.0)

        self.declare_parameter(
            "terminal_status_values",
            ["COMPLETE", "NO_REACHABLE_FRONTIERS", "EXPLORATION_COMPLETE"],
        )

        self.selected_goal_topic = self.get_parameter("selected_goal_topic").value
        self.exploration_status_topic = self.get_parameter("exploration_status_topic").value
        self.navigate_action_name = self.get_parameter("navigate_action_name").value

        self.required_goal_frame = self.get_parameter("required_goal_frame").value
        self.enforce_required_goal_frame = bool(
            self.get_parameter("enforce_required_goal_frame").value
        )

        self.goal_timeout_s = float(self.get_parameter("goal_timeout_s").value)
        self.abort_cooldown_s = float(self.get_parameter("abort_cooldown_s").value)
        self.duplicate_goal_tolerance_m = float(
            self.get_parameter("duplicate_goal_tolerance_m").value
        )
        self.wait_for_server_timeout_s = float(
            self.get_parameter("wait_for_server_timeout_s").value
        )

        self.terminal_status_values = set(self.get_parameter("terminal_status_values").value)

        self.nav_client = ActionClient(self, NavigateToPose, self.navigate_action_name)

        self.active_goal_handle = None
        self.active_goal_pose: Optional[PoseStamped] = None
        self.active_goal_sent_time: Optional[Time] = None

        self.pending_goal: Optional[PoseStamped] = None
        self.last_aborted_time: Optional[Time] = None
        self.stop_requested = False
        self.cancel_in_progress = False

        self.create_subscription(
            PoseStamped,
            self.selected_goal_topic,
            self.selected_goal_callback,
            10,
        )

        self.create_subscription(
            String,
            self.exploration_status_topic,
            self.status_callback,
            10,
        )

        self.create_timer(0.2, self.timer_callback)

        self.get_logger().info(
            "Simple Nav2 executor started:\n"
            f"  selected_goal_topic={self.selected_goal_topic}\n"
            f"  exploration_status_topic={self.exploration_status_topic}\n"
            f"  navigate_action_name={self.navigate_action_name}\n"
            f"  required_goal_frame={self.required_goal_frame}\n"
            f"  goal_timeout_s={self.goal_timeout_s:.1f}\n"
            f"  abort_cooldown_s={self.abort_cooldown_s:.1f}"
        )

    def selected_goal_callback(self, msg: PoseStamped) -> None:
        if self.stop_requested:
            return

        if not msg.header.frame_id:
            self.get_logger().warn("Ignoring selected goal with empty frame_id.")
            return

        if self.enforce_required_goal_frame and msg.header.frame_id != self.required_goal_frame:
            self.get_logger().warn(
                f"Ignoring selected goal in frame '{msg.header.frame_id}', "
                f"required '{self.required_goal_frame}'."
            )
            return

        if self.active_goal_pose is not None:
            if self.same_position(msg, self.active_goal_pose):
                return
            # Minimal baseline: do not replace active goals.
            return

        if self.pending_goal is not None and self.same_position(msg, self.pending_goal):
            return

        self.pending_goal = msg

    def status_callback(self, msg: String) -> None:
        if msg.data in self.terminal_status_values:
            self.stop_requested = True
            self.pending_goal = None
            self.cancel_active_goal()
        else:
            self.stop_requested = False

    def timer_callback(self) -> None:
        now = self.get_clock().now()

        if self.stop_requested:
            return

        if self.active_goal_handle is not None:
            self.check_active_goal_timeout(now)
            return

        if self.pending_goal is None:
            return

        if self.in_abort_cooldown(now):
            return

        if not self.nav_client.server_is_ready():
            available = self.nav_client.wait_for_server(
                timeout_sec=self.wait_for_server_timeout_s
            )
            if not available:
                self.get_logger().warn("NavigateToPose action server unavailable.")
                return

        goal = self.pending_goal
        self.pending_goal = None
        self.send_goal(goal)

    def send_goal(self, pose: PoseStamped) -> None:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.active_goal_pose = goal_msg.pose
        self.active_goal_sent_time = self.get_clock().now()

        self.get_logger().info(
            f"Sending Nav2 goal: x={pose.pose.position.x:.2f}, "
            f"y={pose.pose.position.y:.2f}, frame={pose.header.frame_id}"
        )

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Goal request failed: {exc}")
            self.clear_active_goal()
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected.")
            self.clear_active_goal()
            return

        self.get_logger().info("Nav2 goal accepted.")
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
            self.get_logger().info("Nav2 goal succeeded.")
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn("Nav2 goal aborted. Entering cooldown.")
            self.last_aborted_time = self.get_clock().now()
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("Nav2 goal canceled.")
        elif status is not None:
            self.get_logger().warn(f"Nav2 goal finished with status code {status}.")

        self.clear_active_goal()

    def check_active_goal_timeout(self, now: Time) -> None:
        if self.active_goal_sent_time is None:
            return

        if now - self.active_goal_sent_time > Duration(seconds=self.goal_timeout_s):
            self.get_logger().warn("Nav2 goal timed out. Canceling.")
            self.cancel_active_goal()

    def cancel_active_goal(self) -> None:
        if self.active_goal_handle is None:
            self.clear_active_goal()
            return

        if self.cancel_in_progress:
            return

        self.cancel_in_progress = True
        future = self.active_goal_handle.cancel_goal_async()
        future.add_done_callback(self.cancel_callback)

    def cancel_callback(self, future) -> None:
        try:
            _ = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Cancel failed: {exc}")

        self.cancel_in_progress = False
        self.clear_active_goal()

    def clear_active_goal(self) -> None:
        self.active_goal_handle = None
        self.active_goal_pose = None
        self.active_goal_sent_time = None
        self.cancel_in_progress = False

    def in_abort_cooldown(self, now: Time) -> bool:
        if self.last_aborted_time is None:
            return False

        return now - self.last_aborted_time < Duration(seconds=self.abort_cooldown_s)

    def same_position(self, a: PoseStamped, b: PoseStamped) -> bool:
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y
        return math.hypot(dx, dy) <= self.duplicate_goal_tolerance_m


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationNav2SimpleExecutor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
