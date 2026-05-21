#!/usr/bin/env python3

import json
import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String


class RandomRLPolicySmokeTest(Node):
    def __init__(self):
        super().__init__("random_rl_policy_smoke_test")

        self.observation_topic = "/rl_training_env_stub/observation"
        self.action_topic = "/rl_training_env_stub/action"
        self.status_topic = "/rl_training_env_stub/status"

        self.action_pub = self.create_publisher(Int32, self.action_topic, 10)
        self.create_subscription(String, self.observation_topic, self.observation_callback, 10)
        self.create_subscription(String, self.status_topic, self.status_callback, 10)

        self.latest_status = {}
        self.last_acted_observation_id = None
        self.waiting_for_transition = False
        self.transition_count_at_action = None

        self.get_logger().info(f"Listening: {self.observation_topic}")
        self.get_logger().info(f"Listening: {self.status_topic}")
        self.get_logger().info(f"Publishing: {self.action_topic}")

    def status_callback(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        previous_transition_count = self.latest_status.get("transition_count")
        self.latest_status = status

        current_transition_count = status.get("transition_count")

        if (
            self.waiting_for_transition
            and self.transition_count_at_action is not None
            and current_transition_count is not None
            and int(current_transition_count) > int(self.transition_count_at_action)
        ):
            self.waiting_for_transition = False
            self.transition_count_at_action = None
            self.get_logger().info(
                f"Transition completed. transition_count={current_transition_count}"
            )

    def observation_callback(self, msg: String) -> None:
        if self.waiting_for_transition:
            return

        if bool(self.latest_status.get("active_action", False)):
            return

        if self.latest_status.get("supervisor_state", "") != "EXPLORING":
            return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid observation JSON.")
            return

        observation_id = payload.get("observation_id")
        if observation_id == self.last_acted_observation_id:
            return

        valid_action_mask = payload.get("valid_action_mask", [])
        valid_actions = [
            i for i, value in enumerate(valid_action_mask)
            if int(value) == 1
        ]

        if not valid_actions:
            self.get_logger().warn("No valid actions available.")
            return

        action_index = random.choice(valid_actions)

        out = Int32()
        out.data = int(action_index)
        self.action_pub.publish(out)

        self.last_acted_observation_id = observation_id
        self.waiting_for_transition = True
        self.transition_count_at_action = int(self.latest_status.get("transition_count", 0))

        self.get_logger().info(
            f"observation_id={observation_id}, "
            f"transition_count={self.transition_count_at_action}, "
            f"valid_actions={valid_actions}, "
            f"published_action={action_index}"
        )


def main():
    rclpy.init()
    node = RandomRLPolicySmokeTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
