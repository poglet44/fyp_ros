#!/usr/bin/env python3

from typing import List, Tuple

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
)

from nav_msgs.msg import OccupancyGrid


class OccupancyGridInflater(Node):
    def __init__(self) -> None:
        super().__init__("occupancy_grid_inflater")

        self.declare_parameter("input_topic", "/projected_map")
        self.declare_parameter("output_topic", "/projected_map_inflated")
        self.declare_parameter("inflation_radius_m", 0.7)
        self.declare_parameter("occupied_threshold", 50)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.inflation_radius_m = float(self.get_parameter("inflation_radius_m").value)
        self.occupied_threshold = int(self.get_parameter("occupied_threshold").value)

        # Map-style QoS: transient local so late subscribers still get latest map.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.subscription = self.create_subscription(
            OccupancyGrid,
            self.input_topic,
            self.map_callback,
            qos,
        )

        self.publisher = self.create_publisher(
            OccupancyGrid,
            self.output_topic,
            qos,
        )

        self.get_logger().info(
            f"OccupancyGridInflater started. input_topic='{self.input_topic}', "
            f"output_topic='{self.output_topic}', "
            f"inflation_radius_m={self.inflation_radius_m:.3f}, "
            f"occupied_threshold={self.occupied_threshold}"
        )

    def map_callback(self, msg: OccupancyGrid) -> None:
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        if width == 0 or height == 0:
            self.get_logger().warn("Received empty OccupancyGrid, skipping.")
            return

        if resolution <= 0.0:
            self.get_logger().warn("Received invalid map resolution, skipping.")
            return

        data = np.array(msg.data, dtype=np.int16).reshape((height, width))

        occupied_mask = data >= self.occupied_threshold
        unknown_mask = data < 0
        free_mask = ~occupied_mask & ~unknown_mask

        radius_cells = max(0, int(math.ceil(self.inflation_radius_m / resolution)))
        offsets = self._make_circular_offsets(radius_cells)

        inflated_mask = self._dilate_mask(occupied_mask, offsets)

        # Start from original map.
        out = data.copy()

        # Force original occupied cells to occupied.
        out[occupied_mask] = 100

        # Inflate only into free space. Keep unknown as unknown.
        out[free_mask & inflated_mask] = 100

        out_msg = OccupancyGrid()
        out_msg.header = msg.header
        out_msg.info = msg.info
        out_msg.data = out.astype(np.int8).flatten().tolist()

        self.publisher.publish(out_msg)

    @staticmethod
    def _make_circular_offsets(radius_cells: int) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        r2 = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= r2:
                    offsets.append((dy, dx))
        return offsets

    @staticmethod
    def _dilate_mask(mask: np.ndarray, offsets: List[Tuple[int, int]]) -> np.ndarray:
        height, width = mask.shape
        out = np.zeros_like(mask, dtype=bool)

        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return out

        for y, x in zip(ys, xs):
            for dy, dx in offsets:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    out[ny, nx] = True

        return out


def main(args=None) -> None:
        rclpy.init(args=args)
        node = OccupancyGridInflater()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()