#!/usr/bin/env python3

import argparse
import csv
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.node import Node
from stable_baselines3 import PPO
from std_msgs.msg import Int32, String


class ROSFrontierTrainingNode(Node):
    def __init__(self):
        super().__init__("ppo_frontier_training_bridge")

        self.observation_topic = "/rl_training_env_stub/observation"
        self.status_topic = "/rl_training_env_stub/status"
        self.action_topic = "/rl_training_env_stub/action"

        self.action_pub = self.create_publisher(Int32, self.action_topic, 10)
        self.create_subscription(String, self.observation_topic, self.observation_callback, 10)
        self.create_subscription(String, self.status_topic, self.status_callback, 10)

        self.lock = threading.Lock()
        self.latest_observation_payload: Optional[Dict] = None
        self.latest_status: Optional[Dict] = None
        self.last_observation_time_wall = 0.0
        self.last_status_time_wall = 0.0

    def observation_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid observation JSON.")
            return

        with self.lock:
            self.latest_observation_payload = payload
            self.last_observation_time_wall = time.time()

    def status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid status JSON.")
            return

        with self.lock:
            self.latest_status = payload
            self.last_status_time_wall = time.time()

    def get_observation_payload(self) -> Optional[Dict]:
        with self.lock:
            return self.latest_observation_payload.copy() if self.latest_observation_payload else None

    def get_status(self) -> Optional[Dict]:
        with self.lock:
            return self.latest_status.copy() if self.latest_status else None

    def get_wall_times(self) -> Tuple[float, float]:
        with self.lock:
            return self.last_observation_time_wall, self.last_status_time_wall

    def publish_action(self, action: int) -> None:
        msg = Int32()
        msg.data = int(action)
        self.action_pub.publish(msg)


class FrontierPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        ros_node: ROSFrontierTrainingNode,
        max_episode_time_s: float = 900.0,
        no_status_timeout_s: float = 10.0,
        no_observation_timeout_s: float = 20.0,
        action_completion_timeout_s: float = 180.0,
        recovery_stuck_timeout_s: float = 180.0,
        soft_loop_window: int = 3,
        soft_loop_min_distance_m: float = 10.0,
        soft_loop_max_gain_m2: float = 2.0,
    ):
        super().__init__()

        self.ros_node = ros_node

        self.max_episode_time_s = float(max_episode_time_s)
        self.no_status_timeout_s = float(no_status_timeout_s)
        self.no_observation_timeout_s = float(no_observation_timeout_s)
        self.action_completion_timeout_s = float(action_completion_timeout_s)
        self.recovery_stuck_timeout_s = float(recovery_stuck_timeout_s)

        self.soft_loop_window = int(soft_loop_window)
        self.soft_loop_min_distance_m = float(soft_loop_min_distance_m)
        self.soft_loop_max_gain_m2 = float(soft_loop_max_gain_m2)

        self.max_rl_candidates = 10
        self.feature_count = 12
        self.obs_dim = self.max_rl_candidates * self.feature_count + self.max_rl_candidates

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_rl_candidates)

        self.episode_start_wall = None
        self.recovery_start_wall = None
        self.last_transition_count = 0
        self.last_log_dir = None
        self.last_transition_row_index = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_start_wall = time.time()
        self.recovery_start_wall = None

        obs = self.wait_for_observation(timeout_s=30.0)
        status = self.ros_node.get_status()

        if status is not None:
            self.last_transition_count = int(status.get("transition_count", 0))
            self.last_log_dir = status.get("log_dir")

        return obs, {}

    def step(self, action):
        action = int(action)

        status_before = self.wait_for_status(timeout_s=10.0)
        transition_before = int(status_before.get("transition_count", 0))

        self.ros_node.publish_action(action)

        step_start = time.time()
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "published_action": action,
            "terminal_reason": "",
        }

        while True:
            time.sleep(0.1)

            status = self.ros_node.get_status()
            now = time.time()

            if status is None:
                if now - step_start > self.no_status_timeout_s:
                    truncated = True
                    info["terminal_reason"] = "no_status_timeout"
                    break
                continue

            transition_now = int(status.get("transition_count", 0))
            supervisor_state = str(status.get("supervisor_state", ""))
            active_action = bool(status.get("active_action", False))

            terminal_check = self.check_episode_terminal(status=status)
            if terminal_check is not None:
                terminated, truncated, reason = terminal_check
                info["terminal_reason"] = reason
                reward = self.read_latest_reward(status)
                break

            if transition_now > transition_before:
                row = self.read_latest_transition_row(status)
                if row is not None:
                    reward = float(row.get("reward", 0.0))
                    info.update({
                        "end_reason": row.get("end_reason", ""),
                        "known_area_delta_m2": row.get("known_area_delta_m2", ""),
                        "distance_travelled_m": row.get("distance_travelled_m", ""),
                    })

                soft_loop = self.check_soft_loop(status)
                if soft_loop:
                    truncated = True
                    info["terminal_reason"] = "soft_loop_detected"
                    break

                break

            if now - step_start > self.action_completion_timeout_s:
                truncated = True
                info["terminal_reason"] = "action_completion_timeout"
                reward = -5.0
                break

        obs = self.get_current_or_zero_observation()
        return obs, reward, terminated, truncated, info

    def wait_for_observation(self, timeout_s: float) -> np.ndarray:
        start = time.time()

        while time.time() - start < timeout_s:
            payload = self.ros_node.get_observation_payload()
            status = self.ros_node.get_status()

            if payload is not None and status is not None:
                if str(status.get("supervisor_state", "")) == "EXPLORING":
                    return self.flatten_observation(payload)

            time.sleep(0.1)

        return np.zeros(self.obs_dim, dtype=np.float32)

    def wait_for_status(self, timeout_s: float) -> Dict:
        start = time.time()

        while time.time() - start < timeout_s:
            status = self.ros_node.get_status()
            if status is not None:
                return status
            time.sleep(0.1)

        return {}

    def get_current_or_zero_observation(self) -> np.ndarray:
        payload = self.ros_node.get_observation_payload()
        if payload is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        return self.flatten_observation(payload)

    def flatten_observation(self, payload: Dict) -> np.ndarray:
        observation = payload.get("observation", [])
        valid_action_mask = payload.get("valid_action_mask", [])

        flat: List[float] = []

        for i in range(self.max_rl_candidates):
            if i < len(observation):
                row = observation[i]
            else:
                row = []

            for j in range(self.feature_count):
                try:
                    value = float(row[j]) if j < len(row) else 0.0
                except (TypeError, ValueError):
                    value = 0.0

                if not np.isfinite(value):
                    value = 0.0

                flat.append(value)

        for i in range(self.max_rl_candidates):
            try:
                value = float(valid_action_mask[i]) if i < len(valid_action_mask) else 0.0
            except (TypeError, ValueError):
                value = 0.0

            flat.append(value)

        return np.asarray(flat, dtype=np.float32)

    def check_episode_terminal(self, status: Dict) -> Optional[Tuple[bool, bool, str]]:
        now = time.time()

        if self.episode_start_wall is not None:
            if now - self.episode_start_wall > self.max_episode_time_s:
                return False, True, "max_episode_time"

        obs_time, status_time = self.ros_node.get_wall_times()

        if status_time > 0.0 and now - status_time > self.no_status_timeout_s:
            return False, True, "status_timeout"

        if obs_time > 0.0:
            active_action = bool(status.get("active_action", False))
            supervisor_state = str(status.get("supervisor_state", ""))

            if not active_action and supervisor_state == "EXPLORING":
                if now - obs_time > self.no_observation_timeout_s:
                    return False, True, "observation_timeout"

        supervisor_state = str(status.get("supervisor_state", ""))

        if supervisor_state == "NO_REACHABLE_CANDIDATES_AFTER_RECOVERY":
            return True, False, "normal_no_reachable_candidates"

        if supervisor_state in {"RECOVERY_CHECKPOINT", "RECOVERY_RETURN_START", "RECOVERY_LOCAL_RECHECK"}:
            if self.recovery_start_wall is None:
                self.recovery_start_wall = now
            elif now - self.recovery_start_wall > self.recovery_stuck_timeout_s:
                return False, True, "recovery_stuck_timeout"
        else:
            self.recovery_start_wall = None

        return None

    def read_latest_transition_row(self, status: Dict) -> Optional[Dict[str, str]]:
        log_dir = status.get("log_dir")
        if not log_dir:
            return None

        transitions_path = Path(log_dir) / "transitions.csv"
        if not transitions_path.exists():
            return None

        rows = []
        try:
            with transitions_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            return None

        if not rows:
            return None

        return rows[-1]

    def read_latest_reward(self, status: Dict) -> float:
        row = self.read_latest_transition_row(status)
        if row is None:
            return 0.0

        try:
            return float(row.get("reward", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def check_soft_loop(self, status: Dict) -> bool:
        log_dir = status.get("log_dir")
        if not log_dir:
            return False

        transitions_path = Path(log_dir) / "transitions.csv"
        if not transitions_path.exists():
            return False

        try:
            with transitions_path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return False

        if len(rows) < self.soft_loop_window:
            return False

        recent = rows[-self.soft_loop_window:]

        total_gain = 0.0
        total_distance = 0.0

        for row in recent:
            try:
                total_gain += float(row.get("known_area_delta_m2", 0.0))
                total_distance += float(row.get("distance_travelled_m", 0.0))
            except (TypeError, ValueError):
                return False

        return (
            total_distance >= self.soft_loop_min_distance_m
            and total_gain <= self.soft_loop_max_gain_m2
        )


def spin_ros(node: Node):
    rclpy.spin(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--model-dir", type=str, default="~/Sam/fyp_ws/logs/ppo_models")
    parser.add_argument("--save-name", type=str, default="ppo_frontier_smoke")
    args = parser.parse_args()

    rclpy.init()

    node = ROSFrontierTrainingNode()
    spin_thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    spin_thread.start()

    env = FrontierPPOEnv(node)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=16,
        batch_size=16,
        gamma=0.95,
        learning_rate=3e-4,
        device="auto",
    )

    model.learn(total_timesteps=args.timesteps)

    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / args.save_name
    model.save(str(save_path))

    print(f"Saved PPO model to: {save_path}.zip")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
