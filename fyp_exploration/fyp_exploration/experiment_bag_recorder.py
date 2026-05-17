#!/usr/bin/env python3

import os
import signal
import subprocess
import time
from typing import Optional

import rclpy
from rclpy.node import Node


class ExperimentBagRecorder(Node):
    """
    Wrapper around ros2 bag record.

    Finds the newest active frontier_detector run folder and records selected
    experiment topics into:

        <detector_run_dir>/bag/

    This keeps CSV metrics and bag data inside the same experiment folder.
    """

    def __init__(self):
        super().__init__("experiment_bag_recorder")

        self.declare_parameter("log_root_dir", "~/Sam/fyp_ws/logs")
        self.declare_parameter("use_latest_detector_run_dir", True)
        self.declare_parameter("detector_run_dir_wait_timeout_s", 10.0)
        self.declare_parameter("detector_run_dir_max_age_s", 60.0)

        self.declare_parameter("bag_dir_name", "bag")
        self.declare_parameter("storage_id", "sqlite3")

        self.declare_parameter(
            "topics",
            [
                "/tf",
                "/tf_static",
                "/odom",
                "/exploration_grid",
                "/projected_map",
                "/occupied_cells_vis_array",
                "/selected_frontier_goal",
                "/frontier_path",
                "/frontier_goals",
                "/frontier_cells",
                "/frontier_markers",
                "/frontier_regions_markers",
                "/map_path",
            ],
        )

        self.log_root_dir = os.path.expanduser(
            self.get_parameter("log_root_dir").value
        )
        self.use_latest_detector_run_dir = bool(
            self.get_parameter("use_latest_detector_run_dir").value
        )
        self.detector_run_dir_wait_timeout_s = float(
            self.get_parameter("detector_run_dir_wait_timeout_s").value
        )
        self.detector_run_dir_max_age_s = float(
            self.get_parameter("detector_run_dir_max_age_s").value
        )
        self.bag_dir_name = self.get_parameter("bag_dir_name").value
        self.storage_id = self.get_parameter("storage_id").value
        self.topics = list(self.get_parameter("topics").value)

        self.process: Optional[subprocess.Popen] = None

        self.run_dir = self.resolve_run_dir()
        self.bag_output_dir = os.path.join(self.run_dir, self.bag_dir_name)

        if os.path.exists(self.bag_output_dir):
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.bag_output_dir = os.path.join(
                self.run_dir,
                f"{self.bag_dir_name}_{timestamp}",
            )

        os.makedirs(self.run_dir, exist_ok=True)

        self.start_bag_recording()

    def resolve_run_dir(self) -> str:
        if self.use_latest_detector_run_dir:
            deadline = time.time() + self.detector_run_dir_wait_timeout_s

            while time.time() <= deadline:
                detector_run_dir = self.find_latest_detector_run_dir()

                if detector_run_dir is not None:
                    self.get_logger().info(
                        f"Using latest frontier detector run directory: {detector_run_dir}"
                    )
                    return detector_run_dir

                time.sleep(0.25)

            self.get_logger().warn(
                "Could not find recent frontier_detector run directory. "
                "Falling back to standalone bag directory."
            )

        timestamp = time.strftime("bag_run_%Y-%m-%d_%H-%M-%S")
        return os.path.join(self.log_root_dir, timestamp)

    def find_latest_detector_run_dir(self) -> Optional[str]:
        if not os.path.isdir(self.log_root_dir):
            return None

        now = time.time()
        candidates = []

        for name in os.listdir(self.log_root_dir):
            full_path = os.path.join(self.log_root_dir, name)

            if not os.path.isdir(full_path):
                continue

            if not name.startswith("run_"):
                continue

            expected_files = [
                "run_config.csv",
                "map_metrics.csv",
                "frontier_candidate_metrics.csv",
                "frontier_region_metrics.csv",
            ]

            has_detector_file = any(
                os.path.exists(os.path.join(full_path, filename))
                for filename in expected_files
            )

            if not has_detector_file:
                continue

            modified_time = os.path.getmtime(full_path)
            age_s = now - modified_time

            if age_s > self.detector_run_dir_max_age_s:
                continue

            candidates.append((modified_time, full_path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def start_bag_recording(self):
        command = [
            "ros2",
            "bag",
            "record",
            "-o",
            self.bag_output_dir,
            "--storage",
            self.storage_id,
            "--compression-mode",
            "message",
            "--compression-format",
            "zstd",
        ]

        command.extend(self.topics)

        self.get_logger().info("Starting ros2 bag recorder:")
        self.get_logger().info(" ".join(command))

        self.process = subprocess.Popen(
            command,
            preexec_fn=os.setsid,
        )

        self.get_logger().info(f"Recording bag to: {self.bag_output_dir}")

    def destroy_node(self):
        if self.process is not None and self.process.poll() is None:
            self.get_logger().info("Stopping ros2 bag recorder...")

            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.get_logger().warn(
                    "ros2 bag did not stop after SIGINT. Sending SIGTERM."
                )
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5.0)
            except Exception as exc:
                self.get_logger().error(f"Failed to stop ros2 bag recorder cleanly: {exc}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentBagRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
