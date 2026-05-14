#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.task import Future

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String


class ExplorationNav2Executor(Node):
    """
    Executes selected exploration goals using Nav2 NavigateToPose.

    Inputs:
        /selected_frontier_goal : geometry_msgs/msg/PoseStamped
        /exploration_status     : std_msgs/msg/String

    Backend:
        /navigate_to_pose       : nav2_msgs/action/NavigateToPose
    """

    def __init__(self) -> None:
        super().__init__("exploration_nav2_executor")

        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("exploration_status_topic", "/exploration_status")
        self.declare_parameter("navigate_action_name", "/navigate_to_pose")

        self.declare_parameter("goal_replace_distance_m", 1.25)
        self.declare_parameter("goal_timeout_s", 90.0)
        self.declare_parameter("selected_goal_stale_timeout_s", 5.0)
        self.declare_parameter("wait_for_server_timeout_s", 5.0)

        self.declare_parameter("ignore_goals_without_frame_id", True)
        self.declare_parameter("required_goal_frame", "map")
        self.declare_parameter("enforce_required_goal_frame", True)

        self.declare_parameter("cancel_on_terminal_status", True)
        self.declare_parameter(
            "terminal_status_values",
            ["COMPLETE", "NO_REACHABLE_FRONTIERS", "EXPLORATION_COMPLETE"],
        )
        self.declare_parameter("active_status_values", ["ACTIVE"])

        self.selected_goal_topic = (
            self.get_parameter("selected_goal_topic").get_parameter_value().string_value
        )
        self.exploration_status_topic = (
            self.get_parameter("exploration_status_topic").get_parameter_value().string_value
        )
        self.navigate_action_name = (
            self.get_parameter("navigate_action_name").get_parameter_value().string_value
        )

        self.goal_replace_distance_m = (
            self.get_parameter("goal_replace_distance_m").get_parameter_value().double_value
        )
        self.goal_timeout_s = (
            self.get_parameter("goal_timeout_s").get_parameter_value().double_value
        )
        self.selected_goal_stale_timeout_s = (
            self.get_parameter("selected_goal_stale_timeout_s").get_parameter_value().double_value
        )
        self.wait_for_server_timeout_s = (
            self.get_parameter("wait_for_server_timeout_s").get_parameter_value().double_value
        )

        self.ignore_goals_without_frame_id = (
            self.get_parameter("ignore_goals_without_frame_id").get_parameter_value().bool_value
        )
        self.required_goal_frame = (
            self.get_parameter("required_goal_frame").get_parameter_value().string_value
        )
        self.enforce_required_goal_frame = (
            self.get_parameter("enforce_required_goal_frame").get_parameter_value().bool_value
        )

        self.cancel_on_terminal_status = (
            self.get_parameter("cancel_on_terminal_status").get_parameter_value().bool_value
        )
        self.terminal_status_values = set(
            self.get_parameter("terminal_status_values").get_parameter_value().string_array_value
        )
        self.active_status_values = set(
            self.get_parameter("active_status_values").get_parameter_value().string_array_value
        )

        self.action_client = ActionClient(
            self,
            NavigateToPose,
            self.navigate_action_name,
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.selected_goal_topic,
            self.selected_goal_callback,
            10,
        )

        self.status_sub = self.create_subscription(
            String,
            self.exploration_status_topic,
            self.exploration_status_callback,
            10,
        )

        self.active_goal_pose: Optional[PoseStamped] = None
        self.active_goal_handle = None
        self.active_goal_sent_time = None
        self.last_selected_goal_time = None

        self.pending_goal_pose: Optional[PoseStamped] = None
        self.cancel_in_progress = False
        self.goal_send_in_progress = False
        self.stop_requested_by_status = False

        self.create_timer(0.5, self.timer_callback)

        self.get_logger().info(
            "Exploration Nav2 executor started:\n"
            f"  selected_goal_topic={self.selected_goal_topic}\n"
            f"  exploration_status_topic={self.exploration_status_topic}\n"
            f"  navigate_action_name={self.navigate_action_name}\n"
            f"  goal_replace_distance_m={self.goal_replace_distance_m:.3f}\n"
            f"  goal_timeout_s={self.goal_timeout_s:.3f}\n"
            f"  selected_goal_stale_timeout_s={self.selected_goal_stale_timeout_s:.3f}\n"
            f"  required_goal_frame={self.required_goal_frame}\n"
            f"  enforce_required_goal_frame={self.enforce_required_goal_frame}\n"
            f"  cancel_on_terminal_status={self.cancel_on_terminal_status}"
        )

        self.get_logger().info(
            f"Waiting for Nav2 action server: {self.navigate_action_name}"
        )

        server_ready = self.action_client.wait_for_server(
            timeout_sec=self.wait_for_server_timeout_s
        )

        if server_ready:
            self.get_logger().info("Nav2 action server is available.")
        else:
            self.get_logger().warn(
                "Nav2 action server was not available during startup wait. "
                "The node will keep trying when goals arrive."
            )

    def exploration_status_callback(self, msg: String) -> None:
        status = msg.data.strip().upper()

        if status in self.active_status_values:
            if self.stop_requested_by_status:
                self.get_logger().info(
                    f"Exploration status is {status}. Allowing new navigation goals."
                )
            self.stop_requested_by_status = False
            return

        if status not in self.terminal_status_values:
            return

        if not self.cancel_on_terminal_status:
            return

        self.get_logger().warn(
            f"Received terminal exploration status '{status}'. "
            "Canceling active Nav2 goal and clearing pending goals."
        )

        self.stop_requested_by_status = True
        self.pending_goal_pose = None

        if self.active_goal_handle is not None:
            self.cancel_active_goal()
        else:
            self.clear_active_goal()

    def selected_goal_callback(self, msg: PoseStamped) -> None:
        if not self.goal_is_usable(msg):
            return

        self.last_selected_goal_time = self.get_clock().now()

        if self.stop_requested_by_status:
            self.get_logger().info(
                "Received a new valid selected goal after terminal status. "
                "Clearing stop request."
            )
            self.stop_requested_by_status = False

        if self.active_goal_pose is None and not self.goal_send_in_progress:
            self.get_logger().info(
                "Received first selected exploration goal. Sending to Nav2."
            )
            self.send_goal(msg)
            return

        if self.active_goal_pose is None and self.goal_send_in_progress:
            self.pending_goal_pose = msg
            return

        assert self.active_goal_pose is not None

        distance = self.pose_distance_2d(msg, self.active_goal_pose)

        if distance < self.goal_replace_distance_m:
            self.get_logger().debug(
                f"Ignoring selected goal update: distance from active goal "
                f"{distance:.3f} m < threshold {self.goal_replace_distance_m:.3f} m"
            )
            return

        self.get_logger().info(
            f"New selected goal is {distance:.3f} m from active goal. "
            "Replacing active Nav2 goal."
        )

        self.pending_goal_pose = msg
        self.cancel_active_goal()

    def goal_is_usable(self, goal: PoseStamped) -> bool:
        frame_id = goal.header.frame_id.strip()

        if frame_id == "":
            if self.ignore_goals_without_frame_id:
                self.get_logger().warn("Ignoring selected goal with empty frame_id.")
                return False

        if self.enforce_required_goal_frame and frame_id != self.required_goal_frame:
            self.get_logger().warn(
                f"Ignoring selected goal in frame '{frame_id}'. "
                f"Required frame is '{self.required_goal_frame}'."
            )
            return False

        x = goal.pose.position.x
        y = goal.pose.position.y

        if not math.isfinite(x) or not math.isfinite(y):
            self.get_logger().warn("Ignoring selected goal with non-finite position.")
            return False

        return True

    def send_goal(self, pose: PoseStamped) -> None:
        if self.stop_requested_by_status:
            self.get_logger().warn(
                "Not sending Nav2 goal because terminal exploration status is active."
            )
            return

        if self.goal_send_in_progress:
            self.pending_goal_pose = pose
            return

        if not self.action_client.server_is_ready():
            self.get_logger().warn(
                "Nav2 action server is not ready. Waiting briefly before sending goal."
            )
            ready = self.action_client.wait_for_server(
                timeout_sec=self.wait_for_server_timeout_s
            )
            if not ready:
                self.get_logger().warn(
                    "Nav2 action server is still not ready. Goal not sent."
                )
                return

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose

        self.goal_send_in_progress = True
        self.active_goal_pose = pose

        self.get_logger().info(
            f"Sending Nav2 goal: "
            f"x={pose.pose.position.x:.3f}, "
            f"y={pose.pose.position.y:.3f}, "
            f"frame={pose.header.frame_id}"
        )

        send_future = self.action_client.send_goal_async(
            nav_goal,
            feedback_callback=self.feedback_callback,
        )
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future) -> None:
        self.goal_send_in_progress = False

        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Goal send failed: {exc}")
            self.clear_active_goal()
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 rejected the exploration goal.")
            self.clear_active_goal()
            return

        self.get_logger().info("Nav2 accepted the exploration goal.")

        self.active_goal_handle = goal_handle
        self.active_goal_sent_time = self.get_clock().now()

        if self.stop_requested_by_status:
            self.get_logger().warn(
                "Goal was accepted after terminal status request. Canceling it immediately."
            )
            self.cancel_active_goal()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg) -> None:
        _ = feedback_msg

    def result_callback(self, future: Future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to get Nav2 result: {exc}")
            self.clear_active_goal()
            return

        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Nav2 goal succeeded.")
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn("Nav2 goal aborted.")
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn("Nav2 goal canceled.")
        else:
            self.get_logger().warn(f"Nav2 goal finished with status code: {status}")

        self.clear_active_goal()

        if self.stop_requested_by_status:
            self.pending_goal_pose = None
            return

        if self.pending_goal_pose is not None:
            next_goal = self.pending_goal_pose
            self.pending_goal_pose = None
            self.get_logger().info("Sending pending selected goal after result.")
            self.send_goal(next_goal)

    def cancel_active_goal(self) -> None:
        if self.active_goal_handle is None:
            self.clear_active_goal()
            return

        if self.cancel_in_progress:
            return

        self.cancel_in_progress = True
        cancel_future = self.active_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future: Future) -> None:
        self.cancel_in_progress = False

        try:
            _ = future.result()
        except Exception as exc:
            self.get_logger().error(f"Goal cancel failed: {exc}")

        self.get_logger().info("Active Nav2 goal cancel requested.")

        self.clear_active_goal()

        if self.stop_requested_by_status:
            self.pending_goal_pose = None
            return

        if self.pending_goal_pose is not None:
            next_goal = self.pending_goal_pose
            self.pending_goal_pose = None
            self.send_goal(next_goal)

    def timer_callback(self) -> None:
        if self.active_goal_handle is None:
            return

        now = self.get_clock().now()

        if self.last_selected_goal_time is not None:
            selected_goal_age = now - self.last_selected_goal_time

            if selected_goal_age > Duration(seconds=self.selected_goal_stale_timeout_s):
                self.get_logger().warn(
                    f"No selected exploration goal received for "
                    f"{selected_goal_age.nanoseconds * 1e-9:.1f} s. "
                    "Canceling active Nav2 goal."
                )
                self.pending_goal_pose = None
                self.cancel_active_goal()
                return

        if self.active_goal_sent_time is None:
            return

        elapsed = now - self.active_goal_sent_time

        if elapsed > Duration(seconds=self.goal_timeout_s):
            self.get_logger().warn(
                f"Active Nav2 goal timed out after {elapsed.nanoseconds * 1e-9:.1f} s. "
                "Canceling goal."
            )
            self.cancel_active_goal()

    def clear_active_goal(self) -> None:
        self.active_goal_pose = None
        self.active_goal_handle = None
        self.active_goal_sent_time = None

    @staticmethod
    def pose_distance_2d(a: PoseStamped, b: PoseStamped) -> float:
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y
        return math.hypot(dx, dy)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationNav2Executor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
