#!/usr/bin/env python3

import json
import math
import random
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String


class FrontierCandidateSelector(Node):
    """
    External frontier candidate selector.

    v1 purpose:
      - Subscribe to /frontier_candidates JSON from frontier_detector.
      - Reproduce a simple nearest-frontier selector externally.
      - Publish selected goal to a separate topic for interface testing.

    This node does not modify frontier_detector behaviour.
    """

    RL_FEATURE_KEYS = [
        "path_length_norm",
        "euclidean_distance_norm",
        "path_to_euclidean_ratio_norm",
        "frontier_length_norm",
        "local_unknown_neighbor_area_norm",
        "local_unknown_area_per_path_norm",
        "local_information_density_norm",
        "bearing_to_goal_sin",
        "bearing_to_goal_cos",
        "heading_error_sin",
        "heading_error_cos",
        "stability_cycles_norm",
    ]

    def __init__(self):
        super().__init__("frontier_candidate_selector")

        self.declare_parameter("candidates_topic", "/frontier_candidates")
        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal_external")
        self.declare_parameter("selected_path_topic", "/selected_frontier_path_external")
        self.declare_parameter("status_topic", "/frontier_selector/status")
        self.declare_parameter("selection_policy", "nearest")
        self.declare_parameter("max_candidate_age_s", 2.0)
        self.declare_parameter("random_seed", 0)
        self.declare_parameter("fixed_candidate_index", -1)
        self.declare_parameter("max_rl_candidates", 10)

        self.candidates_topic = str(self.get_parameter("candidates_topic").value)
        self.selected_goal_topic = str(self.get_parameter("selected_goal_topic").value)
        self.selected_path_topic = str(self.get_parameter("selected_path_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.selection_policy = str(self.get_parameter("selection_policy").value)
        self.max_candidate_age_s = float(self.get_parameter("max_candidate_age_s").value)
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.fixed_candidate_index = int(self.get_parameter("fixed_candidate_index").value)
        self.max_rl_candidates = int(self.get_parameter("max_rl_candidates").value)
        self.max_rl_candidates = max(1, self.max_rl_candidates)

        if self.random_seed >= 0:
            self.rng = random.Random(self.random_seed)
        else:
            self.rng = random.Random()

        self.candidate_sub = self.create_subscription(
            String,
            self.candidates_topic,
            self.candidates_callback,
            10,
        )

        self.selected_goal_pub = self.create_publisher(
            PoseStamped,
            self.selected_goal_topic,
            10,
        )

        self.selected_path_pub = self.create_publisher(
            Path,
            self.selected_path_topic,
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            10,
        )

        self.get_logger().info("Frontier candidate selector started.")
        self.get_logger().info(f"Subscribing to: {self.candidates_topic}")
        self.get_logger().info(f"Publishing selected goal to: {self.selected_goal_topic}")
        self.get_logger().info(f"Publishing selected path to: {self.selected_path_topic}")
        self.get_logger().info(f"Selection policy: {self.selection_policy}")
        self.get_logger().info(f"Random seed: {self.random_seed}")
        self.get_logger().info(f"Fixed candidate index: {self.fixed_candidate_index}")
        self.get_logger().info(f"Max RL candidates: {self.max_rl_candidates}")

    def candidates_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.publish_status(
                state="error",
                reason=f"json_decode_error: {exc}",
                payload=None,
                selected_candidate=None,
            )
            return

        candidates = payload.get("candidates", [])

        if not isinstance(candidates, list):
            self.publish_status(
                state="error",
                reason="payload_candidates_not_list",
                payload=payload,
                selected_candidate=None,
            )
            return

        if not candidates:
            self.publish_status(
                state="no_candidate",
                reason="empty_candidate_list",
                payload=payload,
                selected_candidate=None,
            )
            return

        selected = self.select_candidate(candidates)

        if selected is None:
            self.publish_status(
                state="no_valid_candidate",
                reason="no_candidate_satisfied_policy",
                payload=payload,
                selected_candidate=None,
            )
            return

        goal_msg = self.candidate_to_pose_stamped(payload, selected)
        path_msg = self.candidate_to_path(payload, selected)

        self.selected_goal_pub.publish(goal_msg)
        self.selected_path_pub.publish(path_msg)

        self.publish_status(
            state="selected",
            reason="ok",
            payload=payload,
            selected_candidate=selected,
        )

    def select_candidate(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        policy = self.selection_policy.strip().lower()

        valid_candidates = [
            c for c in candidates
            if isinstance(c, dict)
            and c.get("goal_x") is not None
            and c.get("goal_y") is not None
        ]

        if not valid_candidates:
            return None

        if policy == "nearest":
            return self.select_nearest(valid_candidates)

        if policy == "random":
            return self.select_random(valid_candidates)

        if policy == "fixed_index":
            return self.select_fixed_index(valid_candidates)

        if policy == "rl_stub":
            return self.select_rl_stub(valid_candidates)

        self.get_logger().warn(
            f'Unknown selection_policy="{self.selection_policy}". Falling back to nearest.'
        )
        return self.select_nearest(valid_candidates)

    def select_nearest(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        finite = []

        for candidate in candidates:
            path_length = candidate.get("path_length_m", None)

            if path_length is None:
                continue

            try:
                path_length = float(path_length)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(path_length):
                continue

            finite.append((path_length, candidate))

        if not finite:
            return None

        finite.sort(key=lambda item: item[0])
        return finite[0][1]

    def select_random(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        return self.rng.choice(candidates)

    def select_fixed_index(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        for candidate in candidates:
            try:
                candidate_index = int(candidate.get("candidate_index", -1))
            except (TypeError, ValueError):
                continue

            if candidate_index == self.fixed_candidate_index:
                return candidate

        return None

    def finite_path_length(self, candidate: Dict[str, Any]) -> float:
        value = candidate.get("path_length_m", None)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("inf")

        if not math.isfinite(value):
            return float("inf")

        return value

    def preselect_rl_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Deterministic region-balanced preselection.

        Input:
          up to N detector candidates

        Output:
          up to max_rl_candidates candidates for the RL action space

        Rule:
          1. Group candidates by region_id.
          2. Sort candidates inside each region by path_length_m.
          3. Sort regions by nearest candidate path_length_m.
          4. Round-robin through regions:
             nearest from each region first,
             then second-nearest from each region,
             then third-nearest, etc.
        """
        valid_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("goal_x") is not None
            and candidate.get("goal_y") is not None
            and math.isfinite(self.finite_path_length(candidate))
        ]

        if not valid_candidates:
            self.last_rl_preselected_candidate_indices = []
            self.last_rl_valid_action_mask = [0] * self.max_rl_candidates
            return []

        regions: Dict[int, List[Dict[str, Any]]] = {}

        for candidate in valid_candidates:
            try:
                region_id = int(candidate.get("region_id", -1))
            except (TypeError, ValueError):
                region_id = -1

            regions.setdefault(region_id, []).append(candidate)

        for region_candidates in regions.values():
            region_candidates.sort(
                key=lambda candidate: (
                    self.finite_path_length(candidate),
                    int(candidate.get("candidate_index", 10**9)),
                )
            )

        region_ids = sorted(
            regions.keys(),
            key=lambda region_id: (
                self.finite_path_length(regions[region_id][0]),
                region_id,
            )
        )

        selected: List[Dict[str, Any]] = []
        depth = 0

        while len(selected) < self.max_rl_candidates:
            added_this_round = False

            for region_id in region_ids:
                region_candidates = regions[region_id]

                if depth >= len(region_candidates):
                    continue

                selected.append(region_candidates[depth])
                added_this_round = True

                if len(selected) >= self.max_rl_candidates:
                    break

            if not added_this_round:
                break

            depth += 1

        self.last_rl_preselected_candidate_indices = [
            int(candidate.get("candidate_index", -1))
            for candidate in selected
        ]

        self.last_rl_valid_action_mask = (
            [1] * len(selected)
            + [0] * max(0, self.max_rl_candidates - len(selected))
        )

        return selected

    def build_rl_observation(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """
        Build fixed-size RL observation matrix.

        Shape:
          max_rl_candidates x len(RL_FEATURE_KEYS)

        Missing candidates are zero-padded.
        Missing feature values are replaced with 0.0.
        """
        observation: List[List[float]] = []

        for candidate in candidates[: self.max_rl_candidates]:
            row: List[float] = []

            for key in self.RL_FEATURE_KEYS:
                value = candidate.get(key, 0.0)

                if value is None:
                    value = 0.0

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = 0.0

                if not math.isfinite(value):
                    value = 0.0

                row.append(value)

            observation.append(row)

        while len(observation) < self.max_rl_candidates:
            observation.append([0.0] * len(self.RL_FEATURE_KEYS))

        return observation

    def select_rl_stub(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        RL-compatible stub policy.

        This uses the same preselected candidate set and observation format that
        a trained policy will use, but currently chooses the candidate with the
        lowest path_length_norm.
        """
        preselected_candidates = self.preselect_rl_candidates(candidates)

        if not preselected_candidates:
            return None

        _observation = self.build_rl_observation(preselected_candidates)

        best_candidate = None
        best_value = float("inf")

        for candidate in preselected_candidates:
            value = candidate.get("path_length_norm", None)

            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(value):
                continue

            if value < best_value:
                best_value = value
                best_candidate = candidate

        return best_candidate

    def candidate_to_pose_stamped(
        self,
        payload: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> PoseStamped:
        pose = PoseStamped()

        pose.header.frame_id = str(payload.get("map_frame", "map"))

        stamp_sec = int(payload.get("stamp_sec", 0))
        stamp_nanosec = int(payload.get("stamp_nanosec", 0))
        pose.header.stamp.sec = stamp_sec
        pose.header.stamp.nanosec = stamp_nanosec

        goal_x = float(candidate["goal_x"])
        goal_y = float(candidate["goal_y"])
        goal_yaw = float(candidate.get("goal_yaw", 0.0))

        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.25

        pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        pose.pose.orientation.w = math.cos(goal_yaw / 2.0)

        return pose

    def candidate_to_path(
        self,
        payload: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Path:
        path = Path()

        path.header.frame_id = str(payload.get("map_frame", "map"))
        stamp_sec = int(payload.get("stamp_sec", 0))
        stamp_nanosec = int(payload.get("stamp_nanosec", 0))
        path.header.stamp.sec = stamp_sec
        path.header.stamp.nanosec = stamp_nanosec

        path_poses = candidate.get("path_poses", [])
        if not isinstance(path_poses, list):
            path_poses = []

        for item in path_poses:
            if not isinstance(item, dict):
                continue

            try:
                x = float(item["x"])
                y = float(item["y"])
                z = float(item.get("z", 0.05))
            except (KeyError, TypeError, ValueError):
                continue

            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def publish_status(
        self,
        state: str,
        reason: str,
        payload: Optional[Dict[str, Any]],
        selected_candidate: Optional[Dict[str, Any]],
    ) -> None:
        selected_summary = None

        if selected_candidate is not None:
            selected_summary = {
                "candidate_index": selected_candidate.get("candidate_index"),
                "cluster_id": selected_candidate.get("cluster_id"),
                "region_id": selected_candidate.get("region_id"),
                "goal_x": selected_candidate.get("goal_x"),
                "goal_y": selected_candidate.get("goal_y"),
                "path_length_m": selected_candidate.get("path_length_m"),
                "local_unknown_neighbor_area_m2": selected_candidate.get(
                    "local_unknown_neighbor_area_m2"
                ),
            }

        status = {
            "state": state,
            "reason": reason,
            "selection_policy": self.selection_policy,
            "candidate_count": (
                payload.get("candidate_count")
                if isinstance(payload, dict)
                else None
            ),
            "map_count": (
                payload.get("map_count")
                if isinstance(payload, dict)
                else None
            ),
            "selected_candidate": selected_summary,
            "rl_feature_keys": self.RL_FEATURE_KEYS,
            "rl_observation_shape": [
                self.max_rl_candidates,
                len(self.RL_FEATURE_KEYS),
            ],
            "rl_preselected_candidate_indices": getattr(
                self,
                "last_rl_preselected_candidate_indices",
                [],
            ),
            "rl_valid_action_mask": getattr(
                self,
                "last_rl_valid_action_mask",
                [],
            ),
            "output_goal_topic": self.selected_goal_topic,
            "output_path_topic": self.selected_path_topic,
            "selected_path_pose_count": (
                selected_candidate.get("path_pose_count")
                if selected_candidate is not None
                else None
            ),
        }

        msg = String()
        msg.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierCandidateSelector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
