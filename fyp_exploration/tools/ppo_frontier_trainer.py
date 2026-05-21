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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CheckpointCallback
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
        bad_action_min_duration_s: float = 60.0,
        bad_action_min_distance_m: float = 5.0,
        bad_action_max_gain_m2: float = 3.0,
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

        self.bad_action_min_duration_s = float(bad_action_min_duration_s)
        self.bad_action_min_distance_m = float(bad_action_min_distance_m)
        self.bad_action_max_gain_m2 = float(bad_action_max_gain_m2)

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

        print("[PPO_ENV] reset complete", flush=True)
        return obs, {}

    def step(self, action):
        action = int(action)

        status_before = self.wait_for_status(timeout_s=10.0)
        transition_before = int(status_before.get("transition_count", 0))

        # Check the latest valid action mask before publishing.
        # Plain SB3 PPO does not natively support action masks, so it may choose
        # padded actions. Those should be treated as invalid policy actions, not
        # sent to the robot.
        payload = self.ros_node.get_observation_payload()
        valid_action_mask = []
        if payload is not None:
            valid_action_mask = payload.get("valid_action_mask", [])

        action_valid = (
            0 <= action < len(valid_action_mask)
            and int(valid_action_mask[action]) == 1
        )

        if not action_valid:
            print(
                f"[PPO_ENV] invalid padded action={action}, "
                f"valid_action_mask={valid_action_mask}; penalising without publishing.",
                flush=True,
            )
            obs = self.get_current_or_zero_observation()
            info = {
                "published_action": action,
                "terminal_reason": "",
                "end_reason": "invalid_padded_action",
            }
            return obs, -5.0, False, False, info

        print(f"[PPO_ENV] step received action={action}", flush=True)

        accepted = False
        accept_start = time.time()
        publish_count = 0

        while time.time() - accept_start < 5.0:
            status_check = self.ros_node.get_status() or {}

            if bool(status_check.get("active_action", False)):
                accepted = True
                break

            if status_check.get("supervisor_state", "") != "EXPLORING":
                print(
                    "[PPO_ENV] action not sent because supervisor is not EXPLORING: "
                    f"{status_check.get('supervisor_state')}",
                    flush=True,
                )
                break

            if not bool(status_check.get("pending_observation_available", False)):
                print(
                    "[PPO_ENV] waiting for pending observation before sending action",
                    flush=True,
                )
                time.sleep(0.1)
                continue

            self.ros_node.publish_action(action)
            publish_count += 1
            print(
                f"[PPO_ENV] published action={action}, attempt={publish_count}",
                flush=True,
            )
            time.sleep(0.2)

        if not accepted:
            status_check = self.ros_node.get_status() or {}
            print(
                "[PPO_ENV] action was not accepted after repeated publishing. "
                f"active_action={status_check.get('active_action')}, "
                f"pending_observation_available={status_check.get('pending_observation_available')}, "
                f"supervisor_state={status_check.get('supervisor_state')}",
                flush=True,
            )
            obs = self.get_current_or_zero_observation()
            info = {
                "published_action": action,
                "terminal_reason": "action_not_accepted",
                "end_reason": "action_not_accepted",
            }
            return obs, -5.0, False, True, info

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
                print(
                    f"[PPO_ENV] transition completed: before={transition_before}, now={transition_now}",
                    flush=True,
                )
                row = self.read_latest_transition_row(status)
                if row is not None:
                    reward = float(row.get("reward", 0.0))
                    info.update({
                        "end_reason": row.get("end_reason", ""),
                        "known_area_delta_m2": row.get("known_area_delta_m2", ""),
                        "distance_travelled_m": row.get("distance_travelled_m", ""),
                    })

                if row is not None and self.check_single_bad_action(row):
                    truncated = True
                    info["terminal_reason"] = "single_bad_action_detected"
                    print(
                        "[PPO_ENV] single bad action detected; truncating episode. "
                        f"row={row}",
                        flush=True,
                    )
                    break

                soft_loop = self.check_soft_loop(status)
                if soft_loop:
                    truncated = True
                    info["terminal_reason"] = "soft_loop_detected"
                    break

                break

            if now - step_start > self.action_completion_timeout_s:
                print("[PPO_ENV] action completion timeout", flush=True)
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

    def check_single_bad_action(self, row: Dict[str, str]) -> bool:
        """
        Detect one clearly inefficient selected frontier action.

        This does not punish normal terminal recovery-to-start behaviour.
        It only checks the completed RL-selected action row.
        """
        end_reason = str(row.get("end_reason", ""))

        if end_reason not in {"recovery_triggered", "goal_timeout"}:
            return False

        try:
            duration_s = float(row.get("duration_s", 0.0))
            known_gain_m2 = float(row.get("known_area_delta_m2", 0.0))
            distance_m = float(row.get("distance_travelled_m", 0.0))
        except (TypeError, ValueError):
            return False

        return (
            duration_s >= self.bad_action_min_duration_s
            and distance_m >= self.bad_action_min_distance_m
            and known_gain_m2 <= self.bad_action_max_gain_m2
        )

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


def wait_for_active_action_to_finish(
    node: ROSFrontierTrainingNode,
    timeout_s: float = 180.0,
) -> None:
    start = time.time()

    while time.time() - start < timeout_s:
        status = node.get_status()

        if status is None:
            time.sleep(0.2)
            continue

        active_action = bool(status.get("active_action", False))
        supervisor_state = str(status.get("supervisor_state", ""))

        if not active_action:
            print("[PPO_TRAINER] No active action. Safe to exit.", flush=True)
            return

        print(
            "[PPO_TRAINER] Waiting for active robot action to finish before exit... "
            f"active_action={active_action}, "
            f"supervisor_state={supervisor_state}, "
            f"transition_count={status.get('transition_count')}",
            flush=True,
        )
        time.sleep(2.0)

    print(
        "[PPO_TRAINER] Timed out waiting for active action to finish. "
        "Saving anyway; robot stack may continue executing the last goal.",
        flush=True,
    )


def spin_ros(node: Node):
    rclpy.spin(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--model-dir", type=str, default="~/Sam/fyp_ws/logs/ppo_models")
    parser.add_argument("--save-name", type=str, default="ppo_frontier_smoke")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--load-model", type=str, default="")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_path = model_dir / args.save_name

    rclpy.init()

    node = ROSFrontierTrainingNode()
    spin_thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    spin_thread.start()

    env = FrontierPPOEnv(node)

    load_model_path = Path(args.load_model).expanduser() if args.load_model else None

    if load_model_path is not None and load_model_path.exists():
        print(f"Loading existing PPO model: {load_model_path}", flush=True)
        model = PPO.load(str(load_model_path), env=env, device="auto")
    else:
        if load_model_path is not None:
            print(f"Requested load model does not exist yet: {load_model_path}", flush=True)
            print("Starting a new PPO model.", flush=True)

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

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, int(args.checkpoint_every)),
        save_path=str(checkpoint_dir),
        name_prefix=args.save_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current PPO model...")
    finally:
        wait_for_active_action_to_finish(node, timeout_s=180.0)

        model.save(str(save_path))
        print(f"Saved PPO model to: {save_path}.zip")
        print(f"Periodic checkpoints directory: {checkpoint_dir}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
