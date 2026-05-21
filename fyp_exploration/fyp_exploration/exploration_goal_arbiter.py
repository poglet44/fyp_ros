#!/usr/bin/env python3

import copy
import heapq
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


MODE_FRONTIER = "frontier"
MODE_RECOVERY_CHECKPOINT = "recovery_checkpoint"
MODE_RECOVERY_RETURN_START = "recovery_return_start"
MODE_HOLD = "hold"
MODE_NO_VALID_OUTPUT = "no_valid_output"

RECOVERY_STATES = {
    "RECOVERY_CHECKPOINT",
    "RECOVERY_RETURN_START",
}

HOLD_STATES = {
    "RECOVERY_LOCAL_RECHECK",
}


Cell = Tuple[int, int]


class ExplorationGoalArbiter(Node):
    """
    Selects the final exploration goal/path.

    Inputs:
      - raw frontier goal/path from frontier_detector
      - supervisor recovery state/target
      - exploration grid for recovery A*

    Outputs:
      - /selected_exploration_goal
      - /selected_exploration_path
      - /selected_exploration_mode
      - /exploration_goal_arbiter/status
    """

    def __init__(self):
        super().__init__("exploration_goal_arbiter")

        # Inputs
        # normal_goal_topic selects the non-recovery exploration goal source.
        #
        # Baseline mode:
        #   normal_goal_topic: /selected_frontier_goal
        #
        # External/RL selector mode:
        #   normal_goal_topic: /selected_frontier_goal_external
        #
        # frontier_goal_topic is kept for backwards compatibility.
        self.declare_parameter("normal_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("frontier_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("frontier_path_topic", "/frontier_path")
        self.declare_parameter("exploration_grid_topic", "/exploration_grid")
        self.declare_parameter("supervisor_status_topic", "/exploration_supervisor/status")
        self.declare_parameter("recovery_target_goal_topic", "/exploration_supervisor/recovery_target_goal")

        # Outputs
        self.declare_parameter("selected_goal_topic", "/selected_exploration_goal")
        self.declare_parameter("selected_path_topic", "/selected_exploration_path")
        self.declare_parameter("selected_mode_topic", "/selected_exploration_mode")
        self.declare_parameter("status_topic", "/exploration_goal_arbiter/status")

        # Frames / TF
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("tf_lookup_timeout_s", 0.20)

        # Timing
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("frontier_input_timeout_s", 2.0)
        self.declare_parameter("recovery_target_timeout_s", 5.0)
        self.declare_parameter("grid_timeout_s", 5.0)

        # Recovery selection
        self.declare_parameter("enable_recovery_selection", True)
        self.declare_parameter("recovery_replan_period_s", 5.0)

        # Occupancy/path planning
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("free_max_value", 0)
        self.declare_parameter("unknown_is_blocked", True)
        self.declare_parameter("recovery_path_clearance_m", 0.20)
        self.declare_parameter("allow_diagonal_motion", True)
        self.declare_parameter("prevent_diagonal_corner_cutting", True)
        self.declare_parameter("max_astar_expansions", 200000)

        self.normal_goal_topic = self.get_parameter("normal_goal_topic").value

        # Backwards compatibility:
        # internally this is still named frontier_goal_topic, because the rest
        # of the arbiter treats it as the normal non-recovery goal input.
        self.frontier_goal_topic = self.normal_goal_topic

        self.frontier_path_topic = self.get_parameter("frontier_path_topic").value
        self.exploration_grid_topic = self.get_parameter("exploration_grid_topic").value
        self.supervisor_status_topic = self.get_parameter("supervisor_status_topic").value
        self.recovery_target_goal_topic = self.get_parameter("recovery_target_goal_topic").value

        self.selected_goal_topic = self.get_parameter("selected_goal_topic").value
        self.selected_path_topic = self.get_parameter("selected_path_topic").value
        self.selected_mode_topic = self.get_parameter("selected_mode_topic").value
        self.status_topic = self.get_parameter("status_topic").value

        self.map_frame = self.get_parameter("map_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.frontier_input_timeout_s = float(self.get_parameter("frontier_input_timeout_s").value)
        self.recovery_target_timeout_s = float(self.get_parameter("recovery_target_timeout_s").value)
        self.grid_timeout_s = float(self.get_parameter("grid_timeout_s").value)

        self.enable_recovery_selection = bool(self.get_parameter("enable_recovery_selection").value)
        self.recovery_replan_period_s = float(self.get_parameter("recovery_replan_period_s").value)

        self.occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self.free_max_value = int(self.get_parameter("free_max_value").value)
        self.unknown_is_blocked = bool(self.get_parameter("unknown_is_blocked").value)
        self.recovery_path_clearance_m = float(self.get_parameter("recovery_path_clearance_m").value)
        self.allow_diagonal_motion = bool(self.get_parameter("allow_diagonal_motion").value)
        self.prevent_diagonal_corner_cutting = bool(
            self.get_parameter("prevent_diagonal_corner_cutting").value
        )
        self.max_astar_expansions = int(self.get_parameter("max_astar_expansions").value)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.selected_goal_pub = self.create_publisher(PoseStamped, self.selected_goal_topic, 10)
        self.selected_path_pub = self.create_publisher(Path, self.selected_path_topic, 10)
        self.selected_mode_pub = self.create_publisher(String, self.selected_mode_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_subscription(PoseStamped, self.frontier_goal_topic, self.frontier_goal_callback, 10)
        self.create_subscription(Path, self.frontier_path_topic, self.frontier_path_callback, 10)
        self.create_subscription(OccupancyGrid, self.exploration_grid_topic, self.grid_callback, 10)
        self.create_subscription(String, self.supervisor_status_topic, self.supervisor_status_callback, 10)
        self.create_subscription(PoseStamped, self.recovery_target_goal_topic, self.recovery_target_callback, 10)

        self.latest_frontier_goal: Optional[PoseStamped] = None
        self.latest_frontier_goal_time_s: Optional[float] = None

        self.latest_frontier_path: Optional[Path] = None
        self.latest_frontier_path_time_s: Optional[float] = None

        self.latest_grid: Optional[OccupancyGrid] = None
        self.latest_grid_time_s: Optional[float] = None

        self.latest_supervisor_status: Dict = {}
        self.latest_recovery_target: Optional[PoseStamped] = None
        self.latest_recovery_target_time_s: Optional[float] = None

        self.cached_recovery_path: Optional[Path] = None
        self.cached_recovery_target_key: Optional[Tuple[int, int]] = None
        self.cached_recovery_plan_time_s: Optional[float] = None

        self.last_mode = ""
        self.last_reason = ""

        timer_period = 1.0 / max(self.publish_rate_hz, 0.1)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            "exploration_goal_arbiter started:\n"
            f"  normal_goal_topic={self.normal_goal_topic}\n"
            f"  frontier_goal_topic={self.frontier_goal_topic}\n"
            f"  frontier_path_topic={self.frontier_path_topic}\n"
            f"  recovery_target_goal_topic={self.recovery_target_goal_topic}\n"
            f"  selected_goal_topic={self.selected_goal_topic}\n"
            f"  selected_path_topic={self.selected_path_topic}\n"
            f"  enable_recovery_selection={self.enable_recovery_selection}"
        )

    def frontier_goal_callback(self, msg: PoseStamped) -> None:
        self.latest_frontier_goal = msg
        self.latest_frontier_goal_time_s = self.now_seconds()

    def frontier_path_callback(self, msg: Path) -> None:
        self.latest_frontier_path = msg
        self.latest_frontier_path_time_s = self.now_seconds()

    def grid_callback(self, msg: OccupancyGrid) -> None:
        self.latest_grid = msg
        self.latest_grid_time_s = self.now_seconds()

    def supervisor_status_callback(self, msg: String) -> None:
        try:
            self.latest_supervisor_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Ignoring invalid supervisor status JSON.")

    def recovery_target_callback(self, msg: PoseStamped) -> None:
        self.latest_recovery_target = msg
        self.latest_recovery_target_time_s = self.now_seconds()

    def timer_callback(self) -> None:
        supervisor_state = str(self.latest_supervisor_status.get("state", ""))
        recovery_target_type = str(
            self.latest_supervisor_status.get("active_recovery_target_type", "")
        )

        recovery_requested = supervisor_state in RECOVERY_STATES

        if self.enable_recovery_selection and recovery_requested:
            if self.try_publish_recovery_output(supervisor_state, recovery_target_type):
                return

        if supervisor_state in HOLD_STATES:
            self.publish_status(
                mode=MODE_HOLD,
                reason=f"holding_during_{supervisor_state}",
                selected_goal=None,
                selected_path=None,
            )
            return

        self.publish_frontier_output()

    def publish_frontier_output(self) -> None:
        now = self.now_seconds()

        if self.latest_frontier_goal is None or self.latest_frontier_path is None:
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="missing_frontier_goal_or_path",
                selected_goal=None,
                selected_path=None,
            )
            return

        if (
            self.latest_frontier_goal_time_s is None
            or self.latest_frontier_path_time_s is None
            or now - self.latest_frontier_goal_time_s > self.frontier_input_timeout_s
            or now - self.latest_frontier_path_time_s > self.frontier_input_timeout_s
        ):
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="stale_frontier_goal_or_path",
                selected_goal=None,
                selected_path=None,
            )
            return

        goal = copy.deepcopy(self.latest_frontier_goal)
        path = copy.deepcopy(self.latest_frontier_path)

        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()

        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for pose in path.poses:
            pose.header.frame_id = self.map_frame
            pose.header.stamp = path.header.stamp

        self.selected_goal_pub.publish(goal)
        self.selected_path_pub.publish(path)

        self.publish_status(
            mode=MODE_FRONTIER,
            reason="using_frontier_detector_output",
            selected_goal=goal,
            selected_path=path,
        )

    def try_publish_recovery_output(self, supervisor_state: str, recovery_target_type: str) -> bool:
        now = self.now_seconds()

        if self.latest_recovery_target is None:
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_missing_target",
                selected_goal=None,
                selected_path=None,
            )
            return True

        if (
            self.latest_recovery_target_time_s is None
            or now - self.latest_recovery_target_time_s > self.recovery_target_timeout_s
        ):
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_stale_target",
                selected_goal=None,
                selected_path=None,
            )
            return True

        if self.latest_grid is None or self.latest_grid_time_s is None:
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_missing_grid",
                selected_goal=None,
                selected_path=None,
            )
            return True

        if now - self.latest_grid_time_s > self.grid_timeout_s:
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_stale_grid",
                selected_goal=None,
                selected_path=None,
            )
            return True

        robot_cell = self.lookup_robot_cell(self.latest_grid)
        if robot_cell is None:
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_no_robot_pose",
                selected_goal=None,
                selected_path=None,
            )
            return True

        target_cell = self.world_to_cell(
            self.latest_grid,
            self.latest_recovery_target.pose.position.x,
            self.latest_recovery_target.pose.position.y,
        )

        if target_cell is None:
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_target_outside_grid",
                selected_goal=None,
                selected_path=None,
            )
            return True

        target_key = target_cell

        need_replan = (
            self.cached_recovery_path is None
            or self.cached_recovery_target_key != target_key
            or self.cached_recovery_plan_time_s is None
            or now - self.cached_recovery_plan_time_s > self.recovery_replan_period_s
        )

        if need_replan:
            self.cached_recovery_path = self.plan_recovery_path(
                self.latest_grid,
                robot_cell,
                target_cell,
            )
            self.cached_recovery_target_key = target_key
            self.cached_recovery_plan_time_s = now

        if self.cached_recovery_path is None:
            self.publish_status(
                mode=MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_astar_failed",
                selected_goal=None,
                selected_path=None,
            )
            return True

        goal = copy.deepcopy(self.latest_recovery_target)
        path = copy.deepcopy(self.cached_recovery_path)

        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()

        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for pose in path.poses:
            pose.header.frame_id = self.map_frame
            pose.header.stamp = path.header.stamp

        mode = (
            MODE_RECOVERY_RETURN_START
            if supervisor_state == "RECOVERY_RETURN_START" or recovery_target_type == "return_start"
            else MODE_RECOVERY_CHECKPOINT
        )

        self.selected_goal_pub.publish(goal)
        self.selected_path_pub.publish(path)

        self.publish_status(
            mode=mode,
            reason="using_cached_recovery_target_with_astar_path",
            selected_goal=goal,
            selected_path=path,
        )
        return True

    def plan_recovery_path(
        self,
        grid: OccupancyGrid,
        start_cell: Cell,
        goal_cell: Cell,
    ) -> Optional[Path]:
        safe_mask = self.build_safe_mask(grid)

        start = self.find_nearest_safe_cell(safe_mask, start_cell, max_radius_cells=10)
        goal = self.find_nearest_safe_cell(safe_mask, goal_cell, max_radius_cells=20)

        if start is None or goal is None:
            return None

        cells = self.astar(safe_mask, start, goal)
        if not cells:
            return None

        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for cx, cy in cells:
            x, y = self.cell_to_world(grid, cx, cy)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def build_safe_mask(self, grid: OccupancyGrid) -> np.ndarray:
        width = grid.info.width
        height = grid.info.height
        data = np.array(grid.data, dtype=np.int16).reshape((height, width))

        blocked = data >= self.occupied_threshold

        if self.unknown_is_blocked:
            blocked |= data < 0

        blocked |= (data > self.free_max_value) & (data < self.occupied_threshold)

        radius_cells = int(math.ceil(self.recovery_path_clearance_m / grid.info.resolution))
        if radius_cells <= 0:
            return ~blocked

        inflated = blocked.copy()
        obstacle_indices = np.argwhere(blocked)

        for oy, ox in obstacle_indices:
            y0 = max(0, oy - radius_cells)
            y1 = min(height - 1, oy + radius_cells)
            x0 = max(0, ox - radius_cells)
            x1 = min(width - 1, ox + radius_cells)

            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    if math.hypot(xx - ox, yy - oy) <= radius_cells:
                        inflated[yy, xx] = True

        return ~inflated

    def astar(self, safe_mask: np.ndarray, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        height, width = safe_mask.shape

        def in_bounds(cell: Cell) -> bool:
            x, y = cell
            return 0 <= x < width and 0 <= y < height

        def heuristic(a: Cell, b: Cell) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        if not in_bounds(start) or not in_bounds(goal):
            return None

        if not safe_mask[start[1], start[0]] or not safe_mask[goal[1], goal[0]]:
            return None

        if self.allow_diagonal_motion:
            neighbours = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 1, math.sqrt(2.0)),
                (1, -1, math.sqrt(2.0)),
                (-1, 1, math.sqrt(2.0)),
                (-1, -1, math.sqrt(2.0)),
            ]
        else:
            neighbours = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
            ]

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}

        expansions = 0

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            expansions += 1
            if expansions > self.max_astar_expansions:
                return None

            cx, cy = current

            for dx, dy, step_cost in neighbours:
                nx = cx + dx
                ny = cy + dy
                neighbour = (nx, ny)

                if not in_bounds(neighbour):
                    continue

                if not safe_mask[ny, nx]:
                    continue

                if (
                    self.prevent_diagonal_corner_cutting
                    and dx != 0
                    and dy != 0
                    and (
                        not safe_mask[cy, nx]
                        or not safe_mask[ny, cx]
                    )
                ):
                    continue

                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f_score = tentative_g + heuristic(neighbour, goal)
                    heapq.heappush(open_heap, (f_score, neighbour))

        return None

    @staticmethod
    def reconstruct_path(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_nearest_safe_cell(
        self,
        safe_mask: np.ndarray,
        target: Cell,
        max_radius_cells: int,
    ) -> Optional[Cell]:
        height, width = safe_mask.shape
        tx, ty = target

        if 0 <= tx < width and 0 <= ty < height and safe_mask[ty, tx]:
            return target

        best = None
        best_dist = float("inf")

        for radius in range(1, max_radius_cells + 1):
            x0 = max(0, tx - radius)
            x1 = min(width - 1, tx + radius)
            y0 = max(0, ty - radius)
            y1 = min(height - 1, ty + radius)

            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if not safe_mask[y, x]:
                        continue

                    dist = math.hypot(x - tx, y - ty)
                    if dist < best_dist:
                        best_dist = dist
                        best = (x, y)

            if best is not None:
                return best

        return None

    def lookup_robot_cell(self, grid: OccupancyGrid) -> Optional[Cell]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except TransformException:
            return None

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        return self.world_to_cell(grid, x, y)

    def world_to_cell(self, grid: OccupancyGrid, x: float, y: float) -> Optional[Cell]:
        origin = grid.info.origin.position
        resolution = grid.info.resolution

        mx = int(math.floor((x - origin.x) / resolution))
        my = int(math.floor((y - origin.y) / resolution))

        if mx < 0 or my < 0 or mx >= grid.info.width or my >= grid.info.height:
            return None

        return mx, my

    def cell_to_world(self, grid: OccupancyGrid, mx: int, my: int) -> Tuple[float, float]:
        origin = grid.info.origin.position
        resolution = grid.info.resolution

        x = origin.x + (mx + 0.5) * resolution
        y = origin.y + (my + 0.5) * resolution
        return x, y

    def publish_status(
        self,
        mode: str,
        reason: str,
        selected_goal: Optional[PoseStamped],
        selected_path: Optional[Path],
    ) -> None:
        if mode != self.last_mode or reason != self.last_reason:
            self.get_logger().info(f"Arbiter mode={mode}, reason={reason}")
            self.last_mode = mode
            self.last_reason = reason

        mode_msg = String()
        mode_msg.data = mode
        self.selected_mode_pub.publish(mode_msg)

        status = {
            "mode": mode,
            "reason": reason,
            "enable_recovery_selection": self.enable_recovery_selection,

            "supervisor_state": self.latest_supervisor_status.get("state", ""),
            "recovery_target_type": self.latest_supervisor_status.get(
                "active_recovery_target_type", ""
            ),

            "selected_goal_available": selected_goal is not None,
            "selected_path_available": selected_path is not None,
            "selected_goal_x": selected_goal.pose.position.x if selected_goal else None,
            "selected_goal_y": selected_goal.pose.position.y if selected_goal else None,
            "selected_path_pose_count": len(selected_path.poses) if selected_path else 0,

            "normal_goal_topic": self.normal_goal_topic,
            "frontier_goal_topic": self.frontier_goal_topic,
            "frontier_goal_available": self.latest_frontier_goal is not None,
            "frontier_path_available": self.latest_frontier_path is not None,
            "recovery_target_available": self.latest_recovery_target is not None,
            "grid_available": self.latest_grid is not None,

            "map_frame": self.map_frame,
            "robot_frame": self.robot_frame,

            "selected_goal_topic": self.selected_goal_topic,
            "selected_path_topic": self.selected_path_topic,
            "selected_mode_topic": self.selected_mode_topic,
        }

        out = String()
        out.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(out)

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationGoalArbiter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
