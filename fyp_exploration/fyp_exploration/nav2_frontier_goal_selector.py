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


class Nav2FrontierGoalSelector(Node):
    """
    Minimal bridge from /selected_frontier_goal to Nav2 NavigateToPose.

    Responsibility:
      selected frontier goal -> Nav2 action goal

    Not responsible for:
      - frontier detection
      - path planning
      - velocity control
      - recovery behaviour
      - waypoint splitting
    """

    def __init__(self):
        super().__init__("nav2_frontier_goal_selector")

        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("navigate_action_name", "/navigate_to_pose")
        self.declare_parameter("required_goal_frame", "map")
        self.declare_parameter("enforce_required_goal_frame", True)
        self.declare_parameter("duplicate_goal_tolerance_m", 0.25)
        self.declare_parameter("abort_cooldown_s", 5.0)
        self.declare_parameter("wait_for_server_timeout_s", 5.0)

        # Controlled active-goal updating.
        # This allows the frontier detector to change its selected goal while
        # Nav2 is already executing, but prevents cancel/resend spam from small
        # frontier jitter.
        self.declare_parameter("update_active_goal", True)
        self.declare_parameter("update_distance_threshold_m", 0.75)
        self.declare_parameter("min_update_interval_s", 5.0)

        self.selected_goal_topic = self.get_parameter("selected_goal_topic").value
        self.navigate_action_name = self.get_parameter("navigate_action_name").value
        self.required_goal_frame = self.get_parameter("required_goal_frame").value
        self.enforce_required_goal_frame = bool(
            self.get_parameter("enforce_required_goal_frame").value
        )
        self.duplicate_goal_tolerance_m = float(
            self.get_parameter("duplicate_goal_tolerance_m").value
        )
        self.abort_cooldown_s = float(self.get_parameter("abort_cooldown_s").value)
        self.wait_for_server_timeout_s = float(
            self.get_parameter("wait_for_server_timeout_s").value
        )

        self.update_active_goal = bool(self.get_parameter("update_active_goal").value)
        self.update_distance_threshold_m = float(
            self.get_parameter("update_distance_threshold_m").value
        )
        self.min_update_interval_s = float(
            self.get_parameter("min_update_interval_s").value
        )

        self.nav_client = ActionClient(self, NavigateToPose, self.navigate_action_name)

        self.active_goal_handle = None
        self.active_goal_pose: Optional[PoseStamped] = None
        self.pending_goal: Optional[PoseStamped] = None
        self.last_abort_time: Optional[Time] = None
        self.last_goal_sent_time: Optional[Time] = None
        self.canceling_for_update = False

        self.create_subscription(
            PoseStamped,
            self.selected_goal_topic,
            self.selected_goal_callback,
            10,
        )

        self.create_timer(0.2, self.timer_callback)

        self.get_logger().info(
            "nav2_frontier_goal_selector started:\n"
            f"  selected_goal_topic={self.selected_goal_topic}\n"
            f"  navigate_action_name={self.navigate_action_name}\n"
            f"  required_goal_frame={self.required_goal_frame}"
        )

    def selected_goal_callback(self, msg: PoseStamped):
        if not msg.header.frame_id:
            self.get_logger().warn("Ignoring selected goal with empty frame_id.")
            return

        if self.enforce_required_goal_frame and msg.header.frame_id != self.required_goal_frame:
            self.get_logger().warn(
                f"Ignoring goal in frame '{msg.header.frame_id}', "
                f"required '{self.required_goal_frame}'."
            )
            return

        if self.active_goal_pose is not None:
            if self.same_position(msg, self.active_goal_pose):
                return

            if not self.should_update_active_goal(msg):
                return

            self.pending_goal = msg
            self.cancel_active_goal_for_update()
            return

        if self.pending_goal is not None and self.same_position(msg, self.pending_goal):
            return

        self.pending_goal = msg

    def timer_callback(self):
        if self.active_goal_handle is not None:
            return

        if self.pending_goal is None:
            return

        now = self.get_clock().now()
        if self.last_abort_time is not None:
            if now - self.last_abort_time < Duration(seconds=self.abort_cooldown_s):
                return

        if not self.nav_client.server_is_ready():
            ready = self.nav_client.wait_for_server(
                timeout_sec=self.wait_for_server_timeout_s
            )
            if not ready:
                self.get_logger().warn("NavigateToPose action server not available.")
                return

        goal = self.pending_goal
        self.pending_goal = None
        self.send_goal(goal)

    def send_goal(self, pose: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.active_goal_pose = goal_msg.pose
        self.last_goal_sent_time = self.get_clock().now()

        self.get_logger().info(
            f"Sending frontier goal to Nav2: "
            f"x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}"
        )

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Goal request failed: {exc}")
            self.clear_active_goal()
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 rejected frontier goal.")
            self.clear_active_goal()
            return

        self.get_logger().info("Nav2 accepted frontier goal.")
        self.active_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        status = None

        try:
            result = future.result()
            status = result.status
        except Exception as exc:
            self.get_logger().error(f"Failed to get Nav2 result: {exc}")

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Frontier goal succeeded.")
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn("Frontier goal aborted. Entering cooldown.")
            self.last_abort_time = self.get_clock().now()
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("Frontier goal canceled.")
        elif status is not None:
            self.get_logger().warn(f"Frontier goal finished with status code {status}.")

        self.clear_active_goal()

    def should_update_active_goal(self, new_goal: PoseStamped) -> bool:
        if not self.update_active_goal:
            return False

        if self.active_goal_pose is None:
            return False

        if self.canceling_for_update:
            return False

        distance_m = self.goal_distance(new_goal, self.active_goal_pose)

        if distance_m < self.update_distance_threshold_m:
            return False

        now = self.get_clock().now()

        if self.last_goal_sent_time is not None:
            elapsed = now - self.last_goal_sent_time
            if elapsed < Duration(seconds=self.min_update_interval_s):
                return False

        return True

    def cancel_active_goal_for_update(self):
        if self.active_goal_handle is None:
            self.clear_active_goal()
            return

        self.canceling_for_update = True

        self.get_logger().info(
            "Selected frontier goal changed significantly. "
            "Canceling active Nav2 goal for update."
        )

        future = self.active_goal_handle.cancel_goal_async()
        future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        try:
            future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to cancel active Nav2 goal: {exc}")

        self.clear_active_goal()
        self.canceling_for_update = False

    def clear_active_goal(self):
        self.active_goal_handle = None
        self.active_goal_pose = None

    def goal_distance(self, a: PoseStamped, b: PoseStamped) -> float:
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y
        return math.hypot(dx, dy)

    def same_position(self, a: PoseStamped, b: PoseStamped) -> bool:
        return self.goal_distance(a, b) <= self.duplicate_goal_tolerance_m


def main(args=None):
    rclpy.init(args=args)
    node = Nav2FrontierGoalSelector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
