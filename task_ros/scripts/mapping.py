"""Map the workcell with simulated lidar and Gazebo ground truth."""

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

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "map"
RESOLUTION = 0.05


def _floor_bounds():
    """读取当前轴对齐场地的范围；不另维护一份地图尺寸配置。"""
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
    """保存栅格证据；坐标向量按 XY 排列，二维数组按 [y, x] 索引。"""
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
        """将有效激光投影到 world，再转成栅格 XY；返回起点 (2,) 和终点 (N, 2)。"""

        angles = scan.angle_min + np.flatnonzero(usable) * scan.angle_increment
        distances = np.minimum(ranges[usable], scan.range_max)
        rays = np.column_stack((np.cos(angles) * distances, np.sin(angles) * distances))
        q = transform.rotation
        # 水平激光先做完整的三维旋转，再取 world XY；不能忽略传感器安装变换。
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
        """并行离散化射线，只返回地图内的栅格 x、y 索引。"""
        delta = ends - start
        steps = np.abs(delta).max(axis=1)
        ticks = np.arange(steps.max() + 1)
        # 每条射线的采样数不同；零长度射线只保留起点，避免除零。
        fraction = ticks / np.maximum(steps[:, None], 1)
        cells = start + np.rint(delta[:, None, :] * fraction[:, :, None]).astype(int)
        # cells 的形状为 (射线数, 最大采样点数, 2)，最后一维为 XY。
        inside = ((cells >= 0) & (cells < self.size)).all(axis=2)
        return cells[inside & (ticks <= steps[:, None])].T

    def _update_evidence(self, x, y, ends, hits):
        free = np.zeros(self.evidence.shape, dtype=bool)
        free[y, x] = True
        hit_inside = hits & ((ends >= 0) & (ends < self.size)).all(axis=1)
        hit_x, hit_y = ends[hit_inside].T
        # 同一帧中每格只计一次，命中优先于空闲射线。
        free[hit_y, hit_x] = False
        self.evidence[free] -= 1
        self.evidence[hit_y, hit_x] += 3
        # 命中 +3、空闲 -1，证据限制在 [-6, 6]，避免旧观测无限累积。
        np.clip(self.evidence, -6, 6, out=self.evidence)

    def occupancy(self):
        # 足够的命中证据标为占用；负证据为空闲，其余保留未知。
        return np.where(
            self.evidence >= 2, 100, np.where(self.evidence < 0, 0, -1)
        ).astype(np.int8)


class GroundTruthMapping(Node):
    def __init__(self):
        super().__init__(
            "workcell_mapping",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.grid = WorkcellGrid()
        self.received_scan = False
        self._scans = deque(maxlen=16)
        # Use a private TF buffer so the wheel-odometry TF cannot supply the pose.
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
                        "已收到同步的激光和 Gazebo 真值位姿，开始建图。"
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


def _save_map(grid: WorkcellGrid) -> None:
    occupancy = grid.occupancy()[::-1]  # PNG rows run from north to south.
    pixels = np.where(occupancy == 100, 0, np.where(occupancy == 0, 254, 205)).astype(
        np.uint8
    )
    x, y = grid.origin
    OUTPUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(OUTPUT / "map.png")
    # 灰度 205 的占用概率约为 0.196，需高于 free_thresh 才会被读回为未知。
    (OUTPUT / "map.yaml").write_text(
        f"image: map.png\nmode: trinary\nresolution: {RESOLUTION}\n"
        f"origin: [{x:.6f}, {y:.6f}, 0.0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.19\n",
        encoding="utf-8",
    )


def main() -> int:
    # launch 子进程可能继承忽略 SIGINT 的状态，明确使用 Python 中断。
    # ROS 上下文只在 finally 中关闭，避免信号与正在执行的回调争用上下文。
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    mapper = GroundTruthMapping()
    print(
        "等待 /scan 和 /ground_truth/odom。请在另一终端驾驶；按 Ctrl+C 保存地图。",
        flush=True,
    )
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        pass
    finally:
        mapper.destroy_node()
        rclpy.try_shutdown()

    # 栅格是普通 NumPy 数据，ROS 退出后仍可保存；异常则保留完整 traceback。
    if not mapper.received_scan:
        raise RuntimeError("未收到同步的激光和 Gazebo 真值位姿，未保存地图。")
    _save_map(mapper.grid)
    print(f"地图已保存：{OUTPUT / 'map.yaml'} 和 map.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
