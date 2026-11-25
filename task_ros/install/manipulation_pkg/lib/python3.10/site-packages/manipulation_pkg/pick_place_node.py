"""真值辅助的六阶段抓放节点 pick_place_node。

订阅 /robot_description（URDF）、/joint_states（关节反馈）、
/ground_truth/poses（Gazebo 模型世界位姿、链接模型内位姿）、
/ground_truth/odom（底盘速度）和 /navigation_complete（可选的到达事件）。
发布 /gripper_controller/commands（双指力）、/manipulation_state（保留阶段状态）
和 /cmd_vel（仅操作期间发布零速度保持底盘停车）；通过
/arm_controller/follow_joint_trajectory 动作执行七轴轨迹。

物块靠接触摩擦夹持，不设置物块位姿、不添加附着约束。
"""

import signal
import time
from xml.etree import ElementTree

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray, String
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint

from .kinematics import ArmKinematics


def sdf_pose(element):
    values = np.fromstring(element.findtext('pose', '0 0 0 0 0 0'), sep=' ')
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', values[3:]).as_matrix()
    pose[:3, 3] = values[:3]
    return pose


class PickPlace(Node):
    def __init__(self):
        super().__init__(
            'pick_place_node', parameter_overrides=[Parameter('use_sim_time', value=True)]
        )
        defaults = {
            'wait_for_navigation': False,
            'target_model': 'target_block',
            'base_link': 'base_link',
            'tool_link': 'fer_hand_tcp',
            'scene_file': '',
            'approach_height': 0.12,
            'lift_height': 0.15,
            'grasp_offset': 0.012,
            'grasp_pitch': -0.35,
            'joint_speed': 0.35,
            'open_effort': 1.5,
            'grip_effort': 2.0,
            'gripper_timeout': 5.0,
        }
        self.settings = {
            name: self.declare_parameter(name, value).value
            for name, value in defaults.items()
        }
        self.stopping = False
        self.arm = None
        self.description = ''
        self.joints = {}
        self.velocities = {}
        self.poses = {}
        self.odom = None
        self.navigation_complete = False
        self.carrying = False
        self._goal_future = None
        self._goal_handle = None
        self._result_future = None
        self._base_timer = None
        self._pose_names = {
            'mobile_manipulator', 'worktable',
            self.settings['base_link'], self.settings['target_model'],
        }

        world = ElementTree.parse(self.settings['scene_file']).find('world')
        block = world.find(f"model[@name='{self.settings['target_model']}']/link")
        collision = block.find('collision')
        self.block_offset = sdf_pose(block) @ sdf_pose(collision)
        self.block_size = np.fromstring(collision.findtext('geometry/box/size'), sep=' ')
        table = world.find("model[@name='worktable']/link[@name='table_link']")
        tray = table.find("collision[@name='tray_floor_collision']")
        self.tray_offset = sdf_pose(table) @ sdf_pose(tray)
        self.tray_size = np.fromstring(tray.findtext('geometry/box/size'), sep=' ')

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state = self.create_publisher(String, '/manipulation_state', latched)
        self._gripper = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 1
        )
        self._base_command = self.create_publisher(Twist, '/cmd_vel', 1)
        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory'
        )
        self.create_subscription(String, '/robot_description', self._on_description, latched)
        self.create_subscription(JointState, '/joint_states', self._on_joints, qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/ground_truth/poses', self._on_poses, 1)
        self.create_subscription(Odometry, '/ground_truth/odom', self._on_odom, qos_profile_sensor_data)
        self.create_subscription(Empty, '/navigation_complete', self._on_navigation, latched)
        self._stage('waiting', '等待仿真、关节控制器和操作所需真值')

    def _on_description(self, message):
        self.description = message.data

    def _on_joints(self, message):
        self.joints.update(zip(message.name, message.position))
        self.velocities.update(zip(message.name, message.velocity))

    def _on_odom(self, message):
        self.odom = message

    def _on_navigation(self, _message):
        self.navigation_complete = True

    def _on_poses(self, message):
        # SceneBroadcaster 的每个 Pose 没有父坐标系；按其模型/链接语义组合，不能放进 /tf。
        for transform in message.transforms:
            if transform.child_frame_id not in self._pose_names:
                continue
            t, q = transform.transform.translation, transform.transform.rotation
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            pose[:3, 3] = [t.x, t.y, t.z]
            self.poses[transform.child_frame_id] = pose

    def _stage(self, name, detail):
        self._state.publish(String(data=name))
        self.get_logger().info(f'{name}: {detail}')

    def _wait(self, condition, timeout, description):
        # 物理动作的期限使用仿真时间；暂停 Gazebo 不会错误地判定动作超时。
        start = self.get_clock().now()
        while not condition():
            if self.stopping or not rclpy.ok():
                raise InterruptedError('操作已取消')
            if timeout is not None and (self.get_clock().now() - start).nanoseconds > timeout * 1e9:
                raise RuntimeError(f'等待超时：{description}')
            rclpy.spin_once(self, timeout_sec=0.05)

    def _pause(self, seconds):
        until = self.get_clock().now() + Duration(seconds=seconds)
        self._wait(lambda: self.get_clock().now() >= until, seconds + 1.0, '物理状态稳定')

    def _q(self):
        return np.array([self.joints[name] for name in self.arm.joint_names])

    def _base_pose(self):
        return self.poses['mobile_manipulator'] @ self.poses[self.settings['base_link']]

    def _tcp_pose(self):
        # Gazebo 合并了手掌/TCP 固定关节：真实 link7 位姿乘 URDF 固定安装变换。
        return (
            self.poses['mobile_manipulator']
            @ self.poses[self.arm.tool_parent_link]
            @ self.arm.parent_to_tool
        )

    def _block_pose(self):
        return self.poses[self.settings['target_model']] @ self.block_offset

    def _execute(self, waypoints):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.arm.joint_names
        current = self._q()
        elapsed = 0.0
        speed = np.minimum(self.arm.velocity_limits, self.settings['joint_speed'])
        for index, q in enumerate([current, *waypoints]):
            if index:
                # 端点速度/加速度为零，五次插值的速度峰值为 1.875 * 位移 / 时长。
                elapsed += max(0.25, 1.875 * float(np.max(np.abs(q - current) / speed)))
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            point.velocities = [0.0] * len(q)
            point.accelerations = [0.0] * len(q)
            point.time_from_start = Duration(seconds=elapsed).to_msg()
            goal.trajectory.points.append(point)
            current = q

        self._goal_future = self._arm_client.send_goal_async(goal)
        self._wait(self._goal_future.done, 5.0, '机械臂动作接收')
        self._goal_handle = self._goal_future.result()
        if not self._goal_handle.accepted:
            raise RuntimeError('机械臂控制器拒绝轨迹')
        self._result_future = self._goal_handle.get_result_async()
        self._wait(self._result_future.done, elapsed + 3.0, '机械臂轨迹完成')
        response = self._result_future.result()
        if response.status != GoalStatus.STATUS_SUCCEEDED or response.result.error_code != 0:
            raise RuntimeError(
                f'机械臂轨迹失败：status={response.status}, '
                f'code={response.result.error_code}, {response.result.error_string}'
            )
        self._goal_handle = None
        self._goal_future = None
        self._result_future = None
        self._pause(0.15)

    def _move_tcp(self, goal):
        start = self._tcp_pose()
        distance = np.linalg.norm(goal[:3, 3] - start[:3, 3])
        angle = Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).magnitude()
        steps = max(1, int(np.ceil(distance / 0.02)), int(np.ceil(angle / 0.10)))
        rotations = Slerp([0.0, 1.0], Rotation.from_matrix([start[:3, :3], goal[:3, :3]]))
        base_inverse = np.linalg.inv(self._base_pose())
        q = self._q()
        waypoints = []
        for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
            pose = np.eye(4)
            pose[:3, :3] = rotations(float(fraction)).as_matrix()
            pose[:3, 3] = start[:3, 3] + fraction * (goal[:3, 3] - start[:3, 3])
            q = self.arm.solve(base_inverse @ pose, q)
            waypoints.append(q)
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.stopping:
                raise InterruptedError('操作已取消')
        self._execute(waypoints)
        observed = self._tcp_pose()
        position_error = np.linalg.norm(observed[:3, 3] - goal[:3, 3])
        angle_error = Rotation.from_matrix(goal[:3, :3] @ observed[:3, :3].T).magnitude()
        if position_error > 0.008 or angle_error > 0.05:
            raise RuntimeError(
                f'TCP 真值未到位：位置误差 {position_error:.4f} m，姿态误差 {angle_error:.4f} rad'
            )

    def _finger_gap(self):
        return self.joints['fer_finger_joint1'] + self.joints['fer_finger_joint2']

    def _set_gripper(self, opening):
        effort = self.settings['open_effort'] if opening else -self.settings['grip_effort']
        self._gripper.publish(Float64MultiArray(data=[effort, effort]))
        if opening:
            condition = lambda: self._finger_gap() >= 0.078
        else:
            width = self.block_size[1]
            condition = lambda: (
                width * 0.75 <= self._finger_gap() <= width + 0.004
                and max(abs(self.velocities[name]) for name in (
                    'fer_finger_joint1', 'fer_finger_joint2'
                )) < 0.01
            )
        self._wait(condition, self.settings['gripper_timeout'], '夹爪打开' if opening else '物块夹紧')
        self._pause(0.3)
        if not condition():
            raise RuntimeError('夹爪未稳定在预期开口，停止抓放')

    def run(self):
        self._wait(lambda: bool(self.description), None, '机器人描述')
        self.arm = ArmKinematics(
            self.description, self.settings['base_link'], self.settings['tool_link']
        )
        self._pose_names.add(self.arm.tool_parent_link)
        joint_names = [*self.arm.joint_names, 'fer_finger_joint1', 'fer_finger_joint2']
        self._wait(lambda: (
            self._pose_names <= self.poses.keys()
            and all(name in self.joints and name in self.velocities for name in joint_names)
            and self.odom is not None
            and self._arm_client.server_is_ready()
            and self._gripper.get_subscription_count() > 0
            and self._base_command.get_subscription_count() > 0
        ), 30.0, '真值和关节控制器')
        if self.settings['wait_for_navigation']:
            self._stage('waiting', '等待 /navigation_complete，导航未成功不会操作')
            self._wait(lambda: self.navigation_complete, None, '导航成功')
        # 只在导航成功之后接管停车；等待导航期间绝不与导航控制器抢速度话题。
        self._base_timer = self.create_timer(
            0.05, lambda: self._base_command.publish(Twist())
        )
        self._base_command.publish(Twist())
        self._wait(lambda: (
            abs(self.odom.twist.twist.linear.x) < 0.01
            and abs(self.odom.twist.twist.angular.z) < 0.02
        ), 10.0, '底盘停车')
        self._pause(0.4)
        if abs(self.odom.twist.twist.linear.x) >= 0.01 or abs(self.odom.twist.twist.angular.z) >= 0.02:
            raise RuntimeError('底盘尚未停稳，不能操作')
        home = self._q()

        self._stage('grasp', '1/6 打开夹爪，移动到物块上方，再竖直接近')
        self._set_gripper(True)
        block = self._block_pose()
        grasp = block.copy()
        grasp[:3, :3] = (
            block[:3, :3]
            @ Rotation.from_euler('y', self.settings['grasp_pitch']).as_matrix()
            @ Rotation.from_euler('x', np.pi).as_matrix()
        )
        grasp[2, 3] += self.settings['grasp_offset']
        approach = grasp.copy()
        approach[2, 3] += self.settings['approach_height']
        self._move_tcp(approach)
        self._move_tcp(grasp)

        self._stage('close', '2/6 双指力控夹紧，检查物块对应的开口')
        self._set_gripper(False)
        closed_block = self._block_pose().copy()
        self.carrying = True

        self._stage('lift', '3/6 保持夹紧并竖直提升，检查真实物块高度')
        raised = self._tcp_pose().copy()
        raised[2, 3] += self.settings['lift_height']
        self._move_tcp(raised)
        # 允许接触夹持在受重后落座；检验落座后是否稳定，不把夹持当作刚性焊接。
        self._pause(0.5)
        loaded = np.linalg.inv(self._tcp_pose()) @ self._block_pose()
        self._pause(0.3)
        tcp_to_block = np.linalg.inv(self._tcp_pose()) @ self._block_pose()
        shift = np.linalg.norm(tcp_to_block[:3, 3] - loaded[:3, 3])
        turn = Rotation.from_matrix(tcp_to_block[:3, :3] @ loaded[:3, :3].T).magnitude()
        if self._block_pose()[2, 3] - closed_block[2, 3] < 0.75 * self.settings['lift_height']:
            raise RuntimeError('物块未随夹爪提升，抓取失败')
        if shift > 0.003 or turn > 0.05:
            raise RuntimeError('物块在夹爪内持续滑移，停止转移')

        self._stage('place', '4/6 从托盘上方进入，再竖直下降')
        tray = self.poses['worktable'] @ self.tray_offset
        block_goal = tray.copy()
        # 保持实际夹持朝向；最低角点接近托盘底面后松爪，由接触物理自然落稳。
        block_goal[:3, :3] = self._block_pose()[:3, :3]
        relative_rotation = tray[:3, :3].T @ block_goal[:3, :3]
        half_height = np.abs(relative_rotation[2]) @ self.block_size / 2
        block_goal[:3, 3] += tray[:3, 2] * (self.tray_size[2] / 2 + half_height + 0.002)
        place = block_goal @ np.linalg.inv(tcp_to_block)
        above = place.copy()
        above[2, 3] = max(raised[2, 3], place[2, 3] + self.settings['approach_height'])
        self._move_tcp(above)
        self._move_tcp(place)

        self._stage('open', '5/6 松爪并向上退离，确认物块留在托盘')
        self._set_gripper(True)
        self.carrying = False
        retreat = self._tcp_pose().copy()
        retreat[2, 3] += self.settings['approach_height']
        self._move_tcp(retreat)
        self._pause(0.5)
        relative = np.linalg.inv(tray) @ self._block_pose()
        half_extent = np.abs(relative[:3, :3]) @ self.block_size / 2
        clearance = self.tray_size[:2] / 2 - 0.01 - half_extent[:2]
        expected_z = self.tray_size[2] / 2 + half_extent[2]
        if np.any(np.abs(relative[:2, 3]) > clearance) or abs(relative[2, 3] - expected_z) > 0.01:
            raise RuntimeError(f'物块未稳定放入托盘：托盘内坐标 {relative[:3, 3]}')

        self._stage('return', '6/6 机械臂返回抓取前关节姿态')
        self._execute([home])
        if np.max(np.abs(self._q() - home)) > 0.015:
            raise RuntimeError('机械臂未回到抓取前关节姿态')
        self._gripper.publish(Float64MultiArray(data=[0.0, 0.0]))
        self._base_timer.cancel()
        self._base_timer = None
        self._stage('complete', '六阶段完成：物块留在托盘，机械臂已复位')

    def cancel_motion(self):
        """保持 ROS 上下文有效时取消机械臂；已夹住物块时不自动松爪。"""
        if self._base_timer is not None:
            self._base_command.publish(Twist())
        deadline = time.monotonic() + 2.0
        if self._goal_handle is None and self._goal_future is not None:
            while not self._goal_future.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            if self._goal_future.done():
                self._goal_handle = self._goal_future.result()
        if self._goal_handle is not None and self._goal_handle.accepted:
            future = self._goal_handle.cancel_goal_async()
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            if self._result_future is not None:
                while not self._result_future.done() and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.05)
        if not self.carrying:
            self._gripper.publish(Float64MultiArray(data=[0.0, 0.0]))


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PickPlace()
    old_handlers = {}

    def stop(_signum, _frame):
        node.stopping = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, stop)
    try:
        try:
            node.run()
        except (ValueError, RuntimeError) as error:
            node.cancel_motion()
            node._state.publish(String(data='failed'))
            node.get_logger().error(f'抓放失败：{error}')
        while not node.stopping and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except InterruptedError:
        node.get_logger().info('抓放已取消；停止机械臂，保留已有夹持')
    finally:
        node.cancel_motion()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        node.destroy_node()
        rclpy.try_shutdown()
