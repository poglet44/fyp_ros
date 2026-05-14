#!/usr/bin/env python3

from typing import Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import String


class ExplorationStatusMonitor(Node):
    """
    Publishes high-level exploration status.

    Inputs:
        /frontier_goals           geometry_msgs/msg/PoseArray
        /selected_frontier_goal   geometry_msgs/msg/PoseStamped

    Output:
        /exploration_status       std_msgs/msg/String

    Status values:
        WAITING_FOR_GOALS
        ACTIVE
        COMPLETE

    COMPLETE is deliberately conservative:
        - we must have previously seen active exploration
        - no non-empty /frontier_goals for complete_confirm_s
        - no /selected_frontier_goal for selected_goal_silent_confirm_s
    """

    def __init__(self) -> None:
        super().__init__("exploration_status_monitor")

        self.declare_parameter("frontier_goals_topic", "/frontier_goals")
        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("exploration_status_topic", "/exploration_status")

        self.declare_parameter("publish_period_s", 0.5)
        self.declare_parameter("startup_grace_s", 8.0)

        # Conservative completion checks.
        self.declare_parameter("complete_confirm_s", 20.0)
        self.declare_parameter("selected_goal_silent_confirm_s", 20.0)

        self.frontier_goals_topic = (
            self.get_parameter("frontier_goals_topic").get_parameter_value().string_value
        )
        self.selected_goal_topic = (
            self.get_parameter("selected_goal_topic").get_parameter_value().string_value
        )
        self.exploration_status_topic = (
            self.get_parameter("exploration_status_topic").get_parameter_value().string_value
        )
        self.publish_period_s = (
            self.get_parameter("publish_period_s").get_parameter_value().double_value
        )
        self.startup_grace_s = (
            self.get_parameter("startup_grace_s").get_parameter_value().double_value
        )
        self.complete_confirm_s = (
            self.get_parameter("complete_confirm_s").get_parameter_value().double_value
        )
        self.selected_goal_silent_confirm_s = (
            self.get_parameter("selected_goal_silent_confirm_s").get_parameter_value().double_value
        )

        self.status_pub = self.create_publisher(String, self.exploration_status_topic, 10)

        self.frontier_goals_sub = self.create_subscription(
            PoseArray,
            self.frontier_goals_topic,
            self.frontier_goals_callback,
            10,
        )

        self.selected_goal_sub = self.create_subscription(
            PoseStamped,
            self.selected_goal_topic,
            self.selected_goal_callback,
            10,
        )

        now = self.get_clock().now()

        self.start_time = now
        self.last_frontier_goals_msg_time: Optional[rclpy.time.Time] = None
        self.last_nonempty_frontier_goals_time: Optional[rclpy.time.Time] = None
        self.last_selected_goal_time: Optional[rclpy.time.Time] = None

        self.latest_frontier_goal_count: Optional[int] = None
        self.have_seen_active_exploration = False

        self.current_status = "WAITING_FOR_GOALS"
        self.last_published_status: Optional[str] = None

        self.create_timer(self.publish_period_s, self.timer_callback)

        self.get_logger().info(
            "Exploration status monitor started:\n"
            f"  frontier_goals_topic={self.frontier_goals_topic}\n"
            f"  selected_goal_topic={self.selected_goal_topic}\n"
            f"  exploration_status_topic={self.exploration_status_topic}\n"
            f"  startup_grace_s={self.startup_grace_s:.2f}\n"
            f"  complete_confirm_s={self.complete_confirm_s:.2f}\n"
            f"  selected_goal_silent_confirm_s={self.selected_goal_silent_confirm_s:.2f}"
        )

    def frontier_goals_callback(self, msg: PoseArray) -> None:
        now = self.get_clock().now()

        self.last_frontier_goals_msg_time = now
        self.latest_frontier_goal_count = len(msg.poses)

        if self.latest_frontier_goal_count > 0:
            self.last_nonempty_frontier_goals_time = now
            self.have_seen_active_exploration = True
            self.set_status("ACTIVE")

    def selected_goal_callback(self, msg: PoseStamped) -> None:
        _ = msg
        now = self.get_clock().now()

        self.last_selected_goal_time = now
        self.have_seen_active_exploration = True
        self.set_status("ACTIVE")

    def timer_callback(self) -> None:
        now = self.get_clock().now()

        if now - self.start_time < Duration(seconds=self.startup_grace_s):
            if not self.have_seen_active_exploration:
                self.set_status("WAITING_FOR_GOALS")
                self.publish_status_if_needed()
                return

        if not self.have_seen_active_exploration:
            self.set_status("WAITING_FOR_GOALS")
            self.publish_status_if_needed()
            return

        no_recent_nonempty_frontiers = False
        no_recent_selected_goal = False

        if self.last_nonempty_frontier_goals_time is None:
            no_recent_nonempty_frontiers = False
        else:
            frontier_silent_time = now - self.last_nonempty_frontier_goals_time
            no_recent_nonempty_frontiers = frontier_silent_time > Duration(
                seconds=self.complete_confirm_s
            )

        if self.last_selected_goal_time is None:
            no_recent_selected_goal = False
        else:
            selected_silent_time = now - self.last_selected_goal_time
            no_recent_selected_goal = selected_silent_time > Duration(
                seconds=self.selected_goal_silent_confirm_s
            )

        if no_recent_nonempty_frontiers and no_recent_selected_goal:
            self.set_status("COMPLETE")
        else:
            self.set_status("ACTIVE")

        self.publish_status_if_needed()

    def set_status(self, status: str) -> None:
        self.current_status = status

    def publish_status_if_needed(self) -> None:
        if self.current_status == self.last_published_status:
            return

        msg = String()
        msg.data = self.current_status
        self.status_pub.publish(msg)

        self.get_logger().info(f"Exploration status: {self.current_status}")
        self.last_published_status = self.current_status


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationStatusMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
