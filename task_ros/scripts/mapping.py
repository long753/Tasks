"""Map the workcell with simulated lidar and Gazebo ground truth."""

import argparse
import signal
import subprocess
import sys
from array import array
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_prefix
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

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "map"
RESOLUTION = 0.05


def _floor_bounds():
    world = ElementTree.parse(ROOT / "asset" / "workcell.sdf")
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
    def __init__(self):
        self.lower, self.size = _floor_bounds()
        self.origin = self.lower * RESOLUTION
        self.evidence = np.zeros(tuple(self.size[::-1]), dtype=np.int8)

    def insert_scan(self, scan, transform) -> bool:
        ranges = np.asarray(scan.ranges)
        hits = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges < scan.range_max)
        no_return = np.isposinf(ranges) | (ranges == scan.range_max)
        usable = hits | no_return
        if not usable.any():
            return False

        angles = scan.angle_min + np.flatnonzero(usable) * scan.angle_increment
        distances = np.minimum(ranges[usable], scan.range_max)
        rays = np.column_stack((np.cos(angles) * distances, np.sin(angles) * distances))
        q = transform.rotation
        # Rotate the horizontal laser rays in 3D, then take world XY.
        rotation = (
            (1 - 2 * (q.y*q.y + q.z*q.z), 2 * (q.x*q.y - q.w*q.z)),
            (2 * (q.x*q.y + q.w*q.z), 1 - 2 * (q.x*q.x + q.z*q.z)),
        )
        sensor = np.array((transform.translation.x, transform.translation.y))
        start = np.floor(sensor / RESOLUTION).astype(int) - self.lower
        ends = np.floor((sensor + rays @ np.transpose(rotation)) / RESOLUTION).astype(int) - self.lower

        delta = ends - start
        steps = np.abs(delta).max(axis=1)
        ticks = np.arange(steps.max() + 1)
        fraction = ticks / np.maximum(steps[:, None], 1)
        cells = start + np.rint(delta[:, None, :] * fraction[:, :, None]).astype(int)
        inside = ((cells >= 0) & (cells < self.size)).all(axis=2)
        x, y = cells[inside & (ticks <= steps[:, None])].T
        if not x.size:
            return False

        free = np.zeros(self.evidence.shape, dtype=bool)
        free[y, x] = True
        hit_inside = hits[usable] & ((ends >= 0) & (ends < self.size)).all(axis=1)
        hit_x, hit_y = ends[hit_inside].T
        # Count each cell once per scan, with returns taking priority over free rays.
        free[hit_y, hit_x] = False
        self.evidence[free] -= 1
        self.evidence[hit_y, hit_x] += 3
        np.clip(self.evidence, -6, 6, out=self.evidence)
        return True

    def occupancy(self):
        return np.where(self.evidence >= 2, 100, np.where(self.evidence < 0, 0, -1)).astype(np.int8)


class GroundTruthMapping(Node):
    def __init__(self):
        super().__init__("workcell_mapping", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.grid = WorkcellGrid()
        self.received_scan = False
        self._scans = deque(maxlen=16)
        # Use a private TF buffer so the wheel-odometry TF cannot supply the pose.
        self._poses = Buffer(cache_time=Duration(seconds=2))
        latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._map = self.create_publisher(OccupancyGrid, "/map", latched)
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_pose, qos_profile_sensor_data)
        self.create_subscription(TFMessage, "/tf_static", self._on_static, latched)
        self.create_subscription(LaserScan, "/scan", self._scans.append, qos_profile_sensor_data)
        self.create_timer(0.20, self._update_map)

    def _on_pose(self, msg):
        transform = TransformStamped(header=msg.header, child_frame_id=msg.child_frame_id)
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
                pose = self._poses.lookup_transform("world", scan.header.frame_id, Time.from_msg(scan.header.stamp))
            except TransformException:
                continue
            self._scans.clear()
            if self.grid.insert_scan(scan, pose.transform):
                if not self.received_scan:
                    self.get_logger().info("已收到同步的激光和 Gazebo 真值位姿，开始建图。")
                    self.received_scan = True
                self._publish_map(scan.header.stamp)
            break

    def _publish_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = stamp
        msg.info.width, msg.info.height = map(int, self.grid.size)
        msg.info.resolution = RESOLUTION
        msg.info.origin.position.x, msg.info.origin.position.y = map(float, self.grid.origin)
        msg.info.origin.orientation.w = 1.0
        msg.data = array("b", self.grid.occupancy().tobytes())
        self._map.publish(msg)


def _save_map(grid: WorkcellGrid) -> None:
    occupancy = grid.occupancy()[::-1]  # PNG rows run from north to south.
    pixels = np.where(occupancy == 100, 0, np.where(occupancy == 0, 254, 205)).astype(np.uint8)
    x, y = grid.origin
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT) as temp_dir:
        image = Path(temp_dir) / "map.png"
        metadata = image.with_suffix(".yaml")
        Image.fromarray(pixels).save(image)
        # Gray 205 represents unknown; its occupancy (0.196) must exceed free_thresh.
        metadata.write_text(
            f"image: map.png\nmode: trinary\nresolution: {RESOLUTION}\n"
            f"origin: [{x:.6f}, {y:.6f}, 0.0]\nnegate: 0\n"
            "occupied_thresh: 0.65\nfree_thresh: 0.19\n",
            encoding="utf-8",
        )
        image.replace(OUTPUT / image.name)
        metadata.replace(OUTPUT / metadata.name)


def _stop_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rviz", action="store_true", help="Run without the live map window.")
    args = parser.parse_args()

    viewer = None
    if not args.no_rviz:
        try:
            rviz = Path(get_package_prefix("rviz2"))
        except PackageNotFoundError:
            print(
                "请安装 ros-humble-rviz2 以显示地图，或用 --no-rviz 仅建图。",
                file=sys.stderr,
            )
            return 1
        viewer = [
            rviz / "lib/rviz2/rviz2",
            "-d",
            ROOT / "config" / "mapping.rviz",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
        ]

    # Defer shutdown so the last grid remains available for export.
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        with ExitStack() as cleanup:
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            cleanup.callback(rclpy.shutdown)
            mapper = GroundTruthMapping()
            cleanup.callback(mapper.destroy_node)
            if viewer is not None:
                rviz_process = subprocess.Popen(viewer, start_new_session=True)
                cleanup.callback(_stop_process, rviz_process)
            print("等待 /scan 和 /ground_truth/odom。请在另一终端驾驶；按 Ctrl+C 保存地图。", flush=True)
            while not stopping:
                rclpy.spin_once(mapper, timeout_sec=0.25)
            if not mapper.received_scan:
                raise RuntimeError("未收到同步的激光和真值位姿；请用更新后的 main.launch.py 重启仿真。")
            _save_map(mapper.grid)
            print(f"地图已保存：{OUTPUT / 'map.yaml'} 和 map.png", flush=True)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"建图失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
