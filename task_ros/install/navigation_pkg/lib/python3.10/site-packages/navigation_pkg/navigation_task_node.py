"""One-shot Nav2 dock mission ROS node.

ROS node ``navigation_task_node`` consumes localization TF from ``/tf`` and
``/tf_static`` (``map`` to ``base_link``), queries ``/planner_server/get_state``
and, when execution is enabled, ``/controller_server/get_state``.  It publishes
latched ``/goal_pose`` and successful ``/global_plan`` visualizations, sends a
``ComputePathToPose`` goal to ``/compute_path_to_pose``, and optionally sends its
successful path directly to ``/follow_path``.  In execute mode, a successful
``FollowPath`` publishes latched ``std_msgs/msg/Empty`` on
``/navigation_complete``.  It subscribes to no path topic and publishes no
velocity commands.
"""

import signal
import time
from math import cos, sin

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from std_msgs.msg import Empty
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from .scene import marker_pose


class NavigationTask(Node):
    def __init__(self):
        super().__init__(
            "navigation_task_node",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self._execute_path = self.declare_parameter("execute_path", False).value
        self._planner_id = self.declare_parameter("planner_id", "GridBased").value
        self._stopping = False
        self._goal_future = None
        self._result_future = None

        x, y, _, _, _, yaw = marker_pose("dock_marker")
        self._goal = PoseStamped()
        self._goal.header.frame_id = "map"
        self._goal.pose.position.x = x
        self._goal.pose.position.y = y
        self._goal.pose.orientation.z = sin(yaw / 2)
        self._goal.pose.orientation.w = cos(yaw / 2)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", latched)
        self._path_publisher = self.create_publisher(Path, "/global_plan", latched)
        completion_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._navigation_complete_publisher = self.create_publisher(
            Empty, "/navigation_complete", completion_qos
        )
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._planner_state = self.create_client(GetState, "/planner_server/get_state")
        self._planner = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        if self._execute_path:
            self._controller_state = self.create_client(
                GetState, "/controller_server/get_state"
            )
            self._controller = ActionClient(self, FollowPath, "/follow_path")
        self._startup_timer = self.create_timer(0.2, self._wait_for_ready)
        mode = "plan and execute" if self._execute_path else "plan only"
        self.get_logger().info(
            f"Waiting for localization and active Nav2 servers to {mode}: "
            f"dock_marker ({x:.2f}, {y:.2f}), yaw {yaw:.3f} rad."
        )

    def _wait_for_ready(self):
        if self._stopping or not self._tf.can_transform("map", "base_link", Time()):
            return
        if (
            not self._planner.server_is_ready()
            or not self._planner_state.service_is_ready()
        ):
            return
        if self._execute_path and (
            not self._controller.server_is_ready()
            or not self._controller_state.service_is_ready()
        ):
            return

        # Action endpoints exist before their lifecycle nodes are active.
        self._startup_timer.cancel()
        self._planner_state.call_async(GetState.Request()).add_done_callback(
            self._on_planner_state
        )

    def _on_planner_state(self, future):
        if self._stopping:
            return
        if future.result().current_state.id != State.PRIMARY_STATE_ACTIVE:
            self._startup_timer.reset()
            return
        if self._execute_path:
            self._controller_state.call_async(GetState.Request()).add_done_callback(
                self._on_controller_state
            )
            return
        self._start_planning()

    def _on_controller_state(self, future):
        if self._stopping:
            return
        if future.result().current_state.id != State.PRIMARY_STATE_ACTIVE:
            self._startup_timer.reset()
            return
        self._start_planning()

    def _start_planning(self):
        self._goal.header.stamp = self.get_clock().now().to_msg()
        self._goal_publisher.publish(self._goal)
        request = ComputePathToPose.Goal()
        request.goal = self._goal
        request.planner_id = self._planner_id
        request.use_start = False
        self.get_logger().info("Sending one ComputePathToPose goal to dock_marker.")
        self._goal_future = self._planner.send_goal_async(request)
        self._goal_future.add_done_callback(self._on_plan_goal_response)

    def _on_plan_goal_response(self, future):
        if self._stopping:
            return
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("ComputePathToPose rejected dock_marker.")
            return

        self.get_logger().info("ComputePathToPose accepted.")
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._on_plan_result)

    def _on_plan_result(self, future):
        response = future.result()
        if response.status == GoalStatus.STATUS_SUCCEEDED:
            path = response.result.path
            self._path_publisher.publish(path)
            self.get_logger().info(
                f"ComputePathToPose succeeded with {len(path.poses)} poses."
            )
            if self._execute_path and not self._stopping:
                self._start_following(path)
        elif response.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("ComputePathToPose was canceled.")
        else:
            self.get_logger().error(
                f"ComputePathToPose failed with action status {response.status}."
            )

    def _start_following(self, path):
        request = FollowPath.Goal()
        request.path = path
        request.controller_id = "FollowPath"
        request.goal_checker_id = "goal_checker"
        self.get_logger().info("Sending successful path directly to FollowPath.")
        self._result_future = None
        self._goal_future = self._controller.send_goal_async(request)
        self._goal_future.add_done_callback(self._on_follow_goal_response)

    def _on_follow_goal_response(self, future):
        if self._stopping:
            return
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("FollowPath rejected the planned path.")
            return

        self.get_logger().info("FollowPath accepted.")
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._on_follow_result)

    def _on_follow_result(self, future):
        response = future.result()
        if self._execute_path and response.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("FollowPath succeeded; arrived at the dock.")
            self._navigation_complete_publisher.publish(Empty())
        elif response.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("FollowPath was canceled.")
        else:
            self.get_logger().error(
                f"FollowPath failed with action status {response.status}."
            )

    def _wait_for_shutdown(self, future, deadline):
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=max(0.0, deadline - time.monotonic())
        )
        return future.done()

    def cancel_for_shutdown(self):
        """Cancel an active or still-accepting Nav2 action before ROS shutdown."""
        self._stopping = True
        self._startup_timer.cancel()
        if self._goal_future is None:
            return

        deadline = time.monotonic() + 2.0
        if not self._wait_for_shutdown(self._goal_future, deadline):
            self.get_logger().warning("Timed out waiting for Nav2 goal acceptance.")
            return
        goal_handle = self._goal_future.result()
        if not goal_handle.accepted:
            return
        if self._result_future is not None and self._result_future.done():
            return

        self.get_logger().info("Canceling active Nav2 goal before shutdown.")
        if not self._wait_for_shutdown(goal_handle.cancel_goal_async(), deadline):
            self.get_logger().warning("Timed out requesting Nav2 goal cancellation.")
            return
        result = self._result_future or goal_handle.get_result_async()
        if not self._wait_for_shutdown(result, deadline):
            self.get_logger().warning("Timed out confirming Nav2 action stopped.")


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = NavigationTask()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cancel_for_shutdown()
        finally:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
