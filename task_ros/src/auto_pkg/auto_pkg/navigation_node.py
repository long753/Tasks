import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan


class NavigationNode(Node):
    def __init__(self):
        super().__init__("navigation")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.front_distance = None
        self.last_scan = None
        self.create_subscription(
            LaserScan, "/scan", self.on_scan, qos_profile_sensor_data
        )
        self.create_timer(0.1, self.control_step)

    def on_scan(self, msg):
        distances = []
        for i, distance in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) > math.radians(35):
                continue
            if distance == -math.inf:
                distances.append(0.0)  # 近到超出量程
            elif distance == math.inf or msg.range_min <= distance <= msg.range_max:
                distances.append(distance)

        self.front_distance = min(distances, default=None)
        self.last_scan = time.monotonic()

    def control_step(self):
        cmd = Twist()  # 默认零速度
        if (
            self.last_scan is not None
            and time.monotonic() - self.last_scan < 0.5
            and self.front_distance is not None
            and self.front_distance > 0.65
        ):
            cmd.linear.x = 0.10  # m/s
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.cmd_pub.wait_for_all_acked(Duration(seconds=0.2))
        node.destroy_node()
        rclpy.shutdown()
