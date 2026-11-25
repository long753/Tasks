"""节点：moving_obstacle_node，按仿真时间驱动可选移动障碍。

订阅：/clock（启用 use_sim_time，由 ROS 时钟接口接收）。
发布：/workcell/moving_obstacle/set_position（std_msgs/msg/Float64）。
"""

from math import pi, sin

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64


class MovingObstacle(Node):
    def __init__(self):
        super().__init__(
            "moving_obstacle_node",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.publisher = self.create_publisher(
            Float64,
            "/workcell/moving_obstacle/set_position",
            1,
        )
        self.start = None
        self.timer = self.create_timer(0.05, self.update_position)

    def update_position(self):
        now = self.get_clock().now()
        if self.start is None or now < self.start:
            self.start = now
        elapsed = (now - self.start).nanoseconds * 1e-9
        self.publisher.publish(Float64(data=1.1 * sin(2 * pi * elapsed / 12.0)))


def main():
    rclpy.init()
    node = MovingObstacle()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
