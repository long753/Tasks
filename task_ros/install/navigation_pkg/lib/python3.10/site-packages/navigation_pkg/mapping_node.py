"""Ground-truth-assisted (not SLAM) occupancy mapping ROS node.

ROS node ``mapping_node`` subscribes to ``/ground_truth/odom``, ``/tf_static``,
and ``/scan`` and publishes the latched raw occupancy grid on ``/map``.  It uses
Gazebo ground truth plus timestamped laser-link TF only to build the map; it
publishes no velocity commands, services, or actions.
"""

import signal
from array import array
from collections import deque
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from PIL import Image
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException

from .scene import scene_file

RESOLUTION = 0.05


def _floor_bounds():
    """Read the aligned floor extent without maintaining a second map size."""
    world = ElementTree.parse(scene_file())
    floor = world.find("./world/model[@name='floor']")
    collision = floor.find("./link/collision[@name='floor_collision']")
    center = sum(
        np.fromstring(part.findtext("pose", "0 0"), sep=" ", count=2)
        for part in (floor, collision)
    )
    size = np.fromstring(collision.findtext("geometry/box/size"), sep=" ", count=2)
    # Include the outer half of the boundary walls that straddle the floor edges.
    lower = np.floor((center - size / 2) / RESOLUTION).astype(int) - 1
    upper = np.ceil((center + size / 2) / RESOLUTION).astype(int) + 1
    return lower, upper - lower


class WorkcellGrid:
    """Store grid evidence with XY coordinates in an array indexed as ``[y, x]``."""

    def __init__(self):
        self.lower, self.size = _floor_bounds()
        self.origin = self.lower * RESOLUTION
        self.evidence = np.zeros(tuple(self.size[::-1]), dtype=np.int8)

    def insert_scan(self, scan, transform) -> bool:
        ranges = np.asarray(scan.ranges)
        hits = (
            np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges < scan.range_max)
        )
        no_return = np.isposinf(ranges) | (ranges == scan.range_max)
        usable = hits | no_return
        if not usable.any():
            return False
        start, ends = self._project_scan(scan, transform, ranges, usable)
        x, y = self._trace_rays(start, ends)
        if not x.size:
            return False
        self._update_evidence(x, y, ends, hits[usable])
        return True

    def _project_scan(self, scan, transform, ranges, usable):
        """Project valid rays to world coordinates, then map-grid XY coordinates."""
        angles = scan.angle_min + np.flatnonzero(usable) * scan.angle_increment
        distances = np.minimum(ranges[usable], scan.range_max)
        rays = np.column_stack((np.cos(angles) * distances, np.sin(angles) * distances))
        q = transform.rotation
        # Apply the full sensor rotation before retaining horizontal world coordinates.
        rotation = (
            (1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.w * q.z)),
            (2 * (q.x * q.y + q.w * q.z), 1 - 2 * (q.x * q.x + q.z * q.z)),
        )
        sensor = np.array((transform.translation.x, transform.translation.y))
        start = np.floor(sensor / RESOLUTION).astype(int) - self.lower
        ends = (
            np.floor((sensor + rays @ np.transpose(rotation)) / RESOLUTION).astype(int)
            - self.lower
        )
        return start, ends

    def _trace_rays(self, start, ends):
        """Discretize rays in parallel, returning only in-grid ``x, y`` indices."""
        delta = ends - start
        steps = np.abs(delta).max(axis=1)
        ticks = np.arange(steps.max() + 1)
        # Each ray has a distinct length; retain a zero-length ray's start cell.
        fraction = ticks / np.maximum(steps[:, None], 1)
        cells = start + np.rint(delta[:, None, :] * fraction[:, :, None]).astype(int)
        inside = ((cells >= 0) & (cells < self.size)).all(axis=2)
        return cells[inside & (ticks <= steps[:, None])].T

    def _update_evidence(self, x, y, ends, hits):
        free = np.zeros(self.evidence.shape, dtype=bool)
        free[y, x] = True
        hit_inside = hits & ((ends >= 0) & (ends < self.size)).all(axis=1)
        hit_x, hit_y = ends[hit_inside].T
        # Count each cell once per scan and give a hit precedence over free evidence.
        free[hit_y, hit_x] = False
        self.evidence[free] -= 1
        self.evidence[hit_y, hit_x] += 3
        # Hit +3 and free -1 evidence are bounded so stale observations cannot dominate.
        np.clip(self.evidence, -6, 6, out=self.evidence)

    def occupancy(self):
        return np.where(
            self.evidence >= 2, 100, np.where(self.evidence < 0, 0, -1)
        ).astype(np.int8)


class GroundTruthMapping(Node):
    def __init__(self):
        super().__init__(
            "mapping_node",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.output_dir = Path(
            self.declare_parameter(
                "output_dir", str(Path.cwd() / "outputs" / "map")
            ).value
        )
        self.grid = WorkcellGrid()
        self.received_scan = False
        self._scans = deque(maxlen=16)
        # A private buffer prevents wheel-odometry TF from providing the pose.
        self._poses = Buffer(cache_time=Duration(seconds=2))
        latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._map = self.create_publisher(OccupancyGrid, "/map", latched)
        self.create_subscription(
            Odometry, "/ground_truth/odom", self._on_pose, qos_profile_sensor_data
        )
        self.create_subscription(TFMessage, "/tf_static", self._on_static, latched)
        self.create_subscription(
            LaserScan, "/scan", self._scans.append, qos_profile_sensor_data
        )
        self.create_timer(0.20, self._update_map)

    def _on_pose(self, msg):
        transform = TransformStamped(
            header=msg.header, child_frame_id=msg.child_frame_id
        )
        pose = msg.pose.pose
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation = pose.orientation
        self._poses.set_transform(transform, "gazebo")

    def _on_static(self, msg):
        for transform in msg.transforms:
            self._poses.set_transform_static(transform, "robot_state_publisher")

    def _update_map(self):
        for scan in reversed(self._scans):
            try:
                pose = self._poses.lookup_transform(
                    "world", scan.header.frame_id, Time.from_msg(scan.header.stamp)
                )
            except TransformException:
                continue
            self._scans.clear()
            if self.grid.insert_scan(scan, pose.transform):
                if not self.received_scan:
                    self.get_logger().info(
                        "Received synchronized laser and Gazebo ground-truth pose; mapping."
                    )
                    self.received_scan = True
                self._publish_map(scan.header.stamp)
            break

    def _publish_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = stamp
        msg.info.width, msg.info.height = map(int, self.grid.size)
        msg.info.resolution = RESOLUTION
        msg.info.origin.position.x, msg.info.origin.position.y = map(
            float, self.grid.origin
        )
        msg.info.origin.orientation.w = 1.0
        msg.data = array("b", self.grid.occupancy().tobytes())
        self._map.publish(msg)


def _save_map(grid: WorkcellGrid, output_dir: Path) -> None:
    occupancy = grid.occupancy()[::-1]  # PNG rows run from north to south.
    pixels = np.where(occupancy == 100, 0, np.where(occupancy == 0, 254, 205)).astype(
        np.uint8
    )
    x, y = grid.origin
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_dir / "map.png")
    # Gray 205 must read back as unknown rather than free in trinary map_server mode.
    (output_dir / "map.yaml").write_text(
        f"image: map.png\nmode: trinary\nresolution: {RESOLUTION}\n"
        f"origin: [{x:.6f}, {y:.6f}, 0.0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.19\n",
        encoding="utf-8",
    )


def main() -> int:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    mapper = GroundTruthMapping()
    print(
        "Waiting for /scan and /ground_truth/odom; drive in another terminal and press Ctrl+C to save.",
        flush=True,
    )
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        pass
    finally:
        mapper.destroy_node()
        rclpy.try_shutdown()

    if not mapper.received_scan:
        raise RuntimeError(
            "No synchronized laser and Gazebo ground-truth pose received; map not saved."
        )
    _save_map(mapper.grid, mapper.output_dir)
    print(f"Map saved: {mapper.output_dir / 'map.yaml'} and map.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
