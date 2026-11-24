"""Publish a planning-only map with the worktable tabletop marked occupied.

Compose the workcell's horizontal poses, then conservatively bound the rotated
tabletop in map XY. The saved map remains aligned with Gazebo world.
"""

import signal
from array import array
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

ROOT = Path(__file__).resolve().parents[1]
LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


def tabletop_bounds():
    """Read the tabletop's XY bounds, including model, link and collision yaw."""
    world = ElementTree.parse(ROOT / "asset" / "workcell.sdf")
    table = world.find("./world/model[@name='worktable']")
    link = table.find("./link[@name='table_link']")
    tabletop = link.find("./collision[@name='tabletop_collision']")
    center = np.zeros(2)
    rotation = np.eye(2)
    for part in (table, link, tabletop):
        pose = np.fromstring(part.findtext("pose", "0 0 0 0 0 0"), sep=" ")
        center += rotation @ pose[:2]
        c, s = np.cos(pose[5]), np.sin(pose[5])
        rotation = rotation @ np.array(((c, -s), (s, c)))
    size = np.fromstring(tabletop.findtext("geometry/box/size"), sep=" ", count=2)
    # 对 90° 转向是精确矩形；其他平面角度使用保守的轴对齐包围盒。
    half_extent = np.abs(rotation) @ size / 2
    return center - half_extent, center + half_extent


class PlanningMap(Node):
    def __init__(self):
        super().__init__(
            node_name="planning_map",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self._table_lower, self._table_upper = tabletop_bounds()
        self._publisher = self.create_publisher(
            OccupancyGrid, "/planning_map", LATCHED_QOS
        )
        self.create_subscription(OccupancyGrid, "/map", self._on_map, LATCHED_QOS)

    def _on_map(self, source):
        origin = np.array(
            (source.info.origin.position.x, source.info.origin.position.y)
        )
        limits = np.array((source.info.width, source.info.height))
        lower = np.floor((self._table_lower - origin) / source.info.resolution).astype(
            int
        )
        upper = np.ceil((self._table_upper - origin) / source.info.resolution).astype(
            int
        )
        lower = np.clip(lower, 0, limits)
        upper = np.clip(upper, 0, limits)

        data = array("b", source.data)
        grid = np.frombuffer(data, dtype=np.int8).reshape(
            source.info.height, source.info.width
        )
        x0, y0 = lower
        x1, y1 = upper
        grid[y0:y1, x0:x1] = 100

        planning_map = OccupancyGrid()
        planning_map.header = source.header
        planning_map.info = source.info
        planning_map.data = data
        self._publisher.publish(planning_map)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = PlanningMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
