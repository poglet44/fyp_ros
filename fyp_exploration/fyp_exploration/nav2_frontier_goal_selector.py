#!/usr/bin/env python3

import json
import math
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class Nav2FrontierGoalSelector(Node):
    def __init__(self):
        super().__init__("nav2_frontier_goal_selector")

        self.declare_parameter("selected_goal_topic", "/selected_exploration_goal")
        self.declare_parameter("navigate_action_name", "/navigate_to_pose")
        self.declare_parameter("status_topic", "/nav2_frontier_goal_selector/status")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("tf_lookup_timeout_s", 0.20)
        self.declare_parameter("goal_reached_distance_tolerance_m", 0.75)

        self.declare_parameter("required_goal_frame", "map")
        self.declare_parameter("enforce_required_goal_frame", True)

        self.declare_parameter("duplicate_goal_tolerance_m", 0.25)
        self.declare_parameter("abort_cooldown_s", 5.0)
        self.declare_parameter("wait_for_server_timeout_s", 5.0)

        self.declare_parameter("update_active_goal", True)
        self.declare_parameter("update_distance_threshold_m", 0.75)
        self.declare_parameter("min_update_interval_s", 5.0)

        self.selected_goal_topic = self.get_parameter("selected_goal_topic").value
        self.navigate_action_name = self.get_parameter("navigate_action_name").value
        self.status_topic = self.get_parameter("status_topic").value

        self.map_frame = self.get_parameter("map_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)
        self.goal_reached_distance_tolerance_m = float(
            self.get_parameter("goal_reached_distance_tolerance_m").value
        )

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

        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PoseStamped,
            self.selected_goal_topic,
            self.selected_goal_callback,
            10,
        )

        self.pending_goal: Optional[PoseStamped] = None
        self.active_goal_pose: Optional[PoseStamped] = None
        self.active_goal_handle = None

        self.last_goal_sent_time = None
        self.last_abort_time = None
        self.canceling_for_update = False

        self.status_state = "idle"
        self.status_reason = "waiting_for_goal"
        self.last_nav2_status_code = None
        self.last_goal_pose: Optional[PoseStamped] = None
        self.last_goal_dist: Optional[float] = None

        self.create_timer(0.2, self.timer_callback)
        self.create_timer(0.5, self.status_timer_callback)

        self.get_logger().info(
            "Nav2 frontier goal selector started:\n"
            f"  selected_goal_topic={self.selected_goal_topic}\n"
            f"  navigate_action_name={self.navigate_action_name}\n"
            f"  status_topic={self.status_topic}\n"
            f"  required_goal_frame={self.required_goal_frame}"
        )

    def selected_goal_callback(self, msg: PoseStamped):
        if not msg.header.frame_id:
            self.get_logger().warn("Ignoring selected goal with empty frame_id.")
            self.status_state = "idle"
            self.status_reason = "ignored_goal_empty_frame"
            self.publish_executor_status()
            return

        if (
            self.enforce_required_goal_frame
            and msg.header.frame_id != self.required_goal_frame
        ):
            self.get_logger().warn(
                f"Ignoring goal in frame '{msg.header.frame_id}', "
                f"required '{self.required_goal_frame}'."
            )
            self.status_state = "idle"
            self.status_reason = "ignored_goal_wrong_frame"
            self.publish_executor_status()
            return

        if self.active_goal_pose is not None:
            distance = self.pose_distance(msg, self.active_goal_pose)

            if distance < self.duplicate_goal_tolerance_m:
                return

            if self.update_active_goal:
                if not self.can_update_active_goal():
                    return

                if distance >= self.update_distance_threshold_m:
                    self.pending_goal = msg
                    self.cancel_active_goal_for_update()
                    return

        self.pending_goal = msg
        self.status_state = "pending"
        self.status_reason = "pending_goal_received"
        self.publish_executor_status()

    def timer_callback(self):
        if self.pending_goal is None:
            return

        if self.active_goal_handle is not None or self.canceling_for_update:
            return

        if self.last_abort_time is not None:
            since_abort = (self.get_clock().now() - self.last_abort_time).nanoseconds * 1e-9
            if since_abort < self.abort_cooldown_s:
                return

        ready = self.nav_client.wait_for_server(
            timeout_sec=self.wait_for_server_timeout_s
        )

        if not ready:
            self.get_logger().warn("NavigateToPose action server not available.")
            self.status_state = "idle"
            self.status_reason = "nav2_action_server_not_available"
            self.publish_executor_status()
            return

        goal_to_send = self.pending_goal
        self.pending_goal = None
        self.send_goal(goal_to_send)

    def send_goal(self, pose: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.active_goal_pose = pose
        self.last_goal_pose = pose
        self.last_goal_dist = self.compute_goal_distance(pose)
        self.last_goal_sent_time = self.get_clock().now()

        self.status_state = "sending"
        self.status_reason = "sending_goal_to_nav2"
        self.last_nav2_status_code = None
        self.publish_executor_status()

        self.get_logger().info(
            f"Sending Nav2 goal: x={pose.pose.position.x:.2f}, "
            f"y={pose.pose.position.y:.2f}, frame={pose.header.frame_id}"
        )

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Goal request failed: {exc}")
            self.status_state = "failed"
            self.status_reason = "goal_request_failed"
            self.publish_executor_status()
            self.clear_active_goal()
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 rejected frontier goal.")
            self.status_state = "rejected"
            self.status_reason = "goal_rejected"
            self.publish_executor_status()
            self.clear_active_goal()
            return

        self.get_logger().info("Nav2 accepted frontier goal.")

        self.active_goal_handle = goal_handle
        self.status_state = "active"
        self.status_reason = "goal_active"
        self.publish_executor_status()

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        try:
            result = future.result()
            status = result.status
            self.last_nav2_status_code = status
        except Exception as exc:
            self.get_logger().error(f"Failed to get Nav2 result: {exc}")
            self.status_state = "failed"
            self.status_reason = "result_callback_failed"
            self.publish_executor_status()
            self.clear_active_goal()
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.last_goal_dist = self.compute_goal_distance(self.active_goal_pose)
            if (
                self.last_goal_dist is not None
                and self.last_goal_dist <= self.goal_reached_distance_tolerance_m
            ):
                self.get_logger().info("Frontier goal succeeded and distance check passed.")
                self.status_state = "succeeded"
                self.status_reason = "goal_reached"
            else:
                self.get_logger().warn(
                    "Nav2 reported success, but robot is not within goal tolerance. "
                    f"goal_dist={self.last_goal_dist}"
                )
                self.status_state = "succeeded_unverified"
                self.status_reason = "nav2_succeeded_but_goal_distance_large"

        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn("Frontier goal aborted. Entering cooldown.")
            self.last_abort_time = self.get_clock().now()
            self.status_state = "aborted"
            self.status_reason = "goal_aborted"

        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("Frontier goal canceled.")
            self.status_state = "canceled"
            if self.canceling_for_update:
                self.status_reason = "goal_canceled_for_update"
            else:
                self.status_reason = "goal_canceled"

        else:
            self.get_logger().warn(f"Frontier goal finished with status code {status}.")
            self.status_state = "finished"
            self.status_reason = f"finished_status_{status}"

        self.publish_executor_status()
        self.clear_active_goal()

    def cancel_active_goal_for_update(self):
        if self.active_goal_handle is None:
            return

        self.canceling_for_update = True
        self.status_state = "canceling"
        self.status_reason = "canceling_goal_for_update"
        self.publish_executor_status()

        self.get_logger().info("Canceling active Nav2 goal to switch to updated goal.")

        future = self.active_goal_handle.cancel_goal_async()
        future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        try:
            future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to cancel active Nav2 goal: {exc}")
            self.status_state = "failed"
            self.status_reason = "cancel_goal_failed"
            self.publish_executor_status()

        self.clear_active_goal()
        self.canceling_for_update = False

    def clear_active_goal(self):
        self.active_goal_pose = None
        self.active_goal_handle = None
        self.last_goal_sent_time = None

    def can_update_active_goal(self) -> bool:
        if self.last_goal_sent_time is None:
            return True

        dt = (self.get_clock().now() - self.last_goal_sent_time).nanoseconds * 1e-9
        return dt >= self.min_update_interval_s

    def status_timer_callback(self):
        self.publish_executor_status()

    def publish_executor_status(self):
        goal_pose = self.active_goal_pose or self.pending_goal or self.last_goal_pose

        goal_x = None
        goal_y = None
        goal_frame = ""
        goal_dist = None

        if goal_pose is not None:
            goal_x = goal_pose.pose.position.x
            goal_y = goal_pose.pose.position.y
            goal_frame = goal_pose.header.frame_id
            goal_dist = self.compute_goal_distance(goal_pose)

            if goal_dist is not None:
                self.last_goal_dist = goal_dist
            elif self.last_goal_dist is not None:
                goal_dist = self.last_goal_dist

        active_goal_age_s = None
        if self.last_goal_sent_time is not None:
            active_goal_age_s = (
                self.get_clock().now() - self.last_goal_sent_time
            ).nanoseconds * 1e-9

        status = {
            "state": self.status_state,
            "reason": self.status_reason,

            "selected_goal_topic": self.selected_goal_topic,
            "navigate_action_name": self.navigate_action_name,

            "active_goal_available": self.active_goal_pose is not None,
            "pending_goal_available": self.pending_goal is not None,
            "canceling_for_update": bool(self.canceling_for_update),

            "goal_x": goal_x,
            "goal_y": goal_y,
            "goal_frame": goal_frame,
            "goal_dist": goal_dist,

            "map_frame": self.map_frame,
            "robot_frame": self.robot_frame,
            "goal_reached_distance_tolerance_m": self.goal_reached_distance_tolerance_m,

            "active_goal_age_s": active_goal_age_s,
            "last_nav2_status_code": self.last_nav2_status_code,

            "update_active_goal": bool(self.update_active_goal),
            "update_distance_threshold_m": self.update_distance_threshold_m,
            "min_update_interval_s": self.min_update_interval_s,
            "abort_cooldown_s": self.abort_cooldown_s,
        }

        out = String()
        out.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(out)

    def compute_goal_distance(self, goal_pose: Optional[PoseStamped]) -> Optional[float]:
        if goal_pose is None:
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except TransformException:
            return None

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        return math.hypot(
            goal_pose.pose.position.x - robot_x,
            goal_pose.pose.position.y - robot_y,
        )

    @staticmethod
    def pose_distance(a: PoseStamped, b: PoseStamped) -> float:
        return math.hypot(
            a.pose.position.x - b.pose.position.x,
            a.pose.position.y - b.pose.position.y,
        )


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
