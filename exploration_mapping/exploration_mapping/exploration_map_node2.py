import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

import math

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped

import tf2_ros

class ExplorationMapNode(Node):

    def __init__(self):
        super().__init__('exploration_map_node')

        self.publisher = self.create_publisher(
            OccupancyGrid,
            '/exploration_map',
            10
        )
        self.width = 3000
        self.height = 3000
        self.resolution = 0.05
        
                # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Map storage (persist between updates)
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = 'map'
        self.map_msg.info.resolution = self.resolution
        self.map_msg.info.width = self.width
        self.map_msg.info.height = self.height
        self.map_msg.info.origin.position.x = - (self.width * self.resolution) / 2.0
        self.map_msg.info.origin.position.y = - (self.height * self.resolution) / 2.0
        self.map_msg.info.origin.position.z = 0.0
        self.map_msg.info.origin.orientation.w = 1.0
        self.map_msg.data = [-1] * (self.width * self.height)

        # Subscribe to scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
            10
        )

        # Publish at fixed rate (so RViz updates even if scan is slow)
        self.timer = self.create_timer(0.5, self.publish_map)


        
    def publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.map_msg)


    def set_cell(self, x_map: float, y_map: float, value: int):
        # Convert map coords to grid indices
        ox = self.map_msg.info.origin.position.x
        oy = self.map_msg.info.origin.position.y
        res = self.map_msg.info.resolution

        gx = int((x_map - ox) / res)
        gy = int((y_map - oy) / res)

        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return

        idx = gy * self.width + gx
        current = self.map_msg.data[idx]
        if current == 100 and value == 0:
            return  # don't erase obstacles
        self.map_msg.data[idx] = value
        
    def scan_cb(self, scan: LaserScan):
        # Get transform map -> base_link
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                'map',
                scan.header.frame_id,   # often 'base_scan' or similar
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # Extract translation (we'll ignore rotation for now ONLY if scan frame aligned; better handle yaw)
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y

        # Extract yaw from quaternion
        q = tf.transform.rotation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        angle = scan.angle_min
        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue
            if r < scan.range_min or r > scan.range_max:
                angle += scan.angle_increment
                continue

            # hit point in scan frame (2D)
            xs = r * math.cos(angle)
            ys = r * math.sin(angle)

            # ---- FREE SPACE CARVING ----
            step = self.resolution  # step size along the ray
            s = 0.0
            while s < r:
                xs_f = s * math.cos(angle)
                ys_f = s * math.sin(angle)

                xm_f = tx + (xs_f * math.cos(yaw) - ys_f * math.sin(yaw))
                ym_f = ty + (xs_f * math.sin(yaw) + ys_f * math.cos(yaw))

                self.set_cell(xm_f, ym_f, 0)
                s += step

            # ---- OCCUPIED ENDPOINT ----
            xm = tx + (xs * math.cos(yaw) - ys * math.sin(yaw))
            ym = ty + (xs * math.sin(yaw) + ys * math.cos(yaw))
            self.set_cell(xm, ym, 100)

            angle += scan.angle_increment
            
                    
    def publish_blank_map(self):

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height

        msg.info.origin.position.x = - (self.width * self.resolution) / 2.0
        msg.info.origin.position.y = - (self.height * self.resolution) / 2.0
        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.w = 1.0

        msg.data = [-1] * (self.width * self.height)

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()