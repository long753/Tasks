"""Plan once to the workcell dock after AMCL and Nav2 are ready; never drive."""

import signal
from itertools import pairwise
from math import cos, hypot, sin
from pathlib import Path
from xml.etree import ElementTree

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path as NavPath
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

ROOT = Path(__file__).resolve().parents[1]


class GoalPlanner(Node):
    def __init__(self):
        super().__init__(
            "goal_planner",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        # 与场景共用最终目标；地图 XY 与 Gazebo world 对齐，只取平面位姿。
        world = ElementTree.parse(ROOT / "asset" / "workcell.sdf")
        x, y, _, _, _, yaw = map(
            float, world.findtext("./world/model[@name='dock_marker']/pose").split()
        )
        self._goal = PoseStamped()
        self._goal.header.frame_id = "map"
        self._goal.pose.position.x = x
        self._goal.pose.position.y = y
        self._goal.pose.orientation.z = sin(yaw / 2)
        self._goal.pose.orientation.w = cos(yaw / 2)

        # 自动规划可能早于 RViz 启动，保留目标和结果供晚加入的订阅者读取。
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", latched)
        self._path_publisher = self.create_publisher(NavPath, "/global_plan", latched)
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._state_client = self.create_client(GetState, "/planner_server/get_state")
        self._planner = ActionClient(
            self,
            ComputePathToPose,
            "/compute_path_to_pose",
        )
        self._startup_timer = self.create_timer(0.2, self._wait_for_ready)
        self.get_logger().info(
            f"Waiting for AMCL and planner; dock_marker target: "
            f"({x:.2f}, {y:.2f}), yaw {yaw:.3f} rad."
        )

    def _wait_for_ready(self):
        if not self._tf.can_transform("map", "base_link", Time()):
            return
        if not self._planner.server_is_ready() or not self._state_client.service_is_ready():
            return

        # Action 服务在配置阶段就会出现，必须等规划器进入 active 状态再发目标。
        self._startup_timer.cancel()
        self._state_client.call_async(GetState.Request()).add_done_callback(
            self._on_planner_state
        )

    def _on_planner_state(self, future):
        if future.result().current_state.id != State.PRIMARY_STATE_ACTIVE:
            self._startup_timer.reset()
            return

        self._goal.header.stamp = self.get_clock().now().to_msg()
        self._goal_publisher.publish(self._goal)
        self.get_logger().info("Automatically planning to dock_marker.")
        request = ComputePathToPose.Goal()
        request.goal = self._goal
        request.planner_id = "GridBased"
        request.use_start = False
        future = self._planner.send_goal_async(request)
        future.add_done_callback(self._goal_response)

    def _goal_response(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                "ComputePathToPose rejected dock_marker."
            )
            return

        self.get_logger().info("ComputePathToPose goal accepted.")
        goal_handle.get_result_async().add_done_callback(self._plan_result)

    def _plan_result(self, future):
        response = future.result()

        if response.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"ComputePathToPose failed with action status {response.status}."
            )
            return

        path = response.result.path
        self._path_publisher.publish(path)
        length = sum(
            hypot(
                b.pose.position.x - a.pose.position.x,
                b.pose.position.y - a.pose.position.y,
            )
            for a, b in pairwise(path.poses)
        )
        planning_time = response.result.planning_time
        seconds = planning_time.sec + planning_time.nanosec * 1e-9
        self.get_logger().info(
            f"A* path: {len(path.poses)} poses, {length:.2f} m, computed in {seconds:.3f} s."
        )


def main():
    # 与建图节点一致：让 Python 中断退出，最后统一关闭 ROS 上下文。
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = GoalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
