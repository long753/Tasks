"""从展开的 URDF 读取七轴运动链，计算 base_link → TCP 的 FK 和有界数值 IK。"""

from xml.etree import ElementTree

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _origin(joint):
    element = joint.find('origin')
    pose = np.eye(4)
    if element is not None:
        pose[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
        pose[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    return pose


class ArmKinematics:
    """只接受 robot_description 的展开 XML，不另维护 DH 参数或加载 xacro。"""

    def __init__(self, urdf, base_link='base_link', tool_link='fer_hand_tcp'):
        robot = ElementTree.fromstring(urdf)
        by_child = {
            joint.find('child').get('link'): joint
            for joint in robot.findall('joint')
        }
        chain = []
        link = tool_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f'URDF 中没有 {base_link} → {tool_link} 运动链')
            joint = by_child[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        chain.reverse()

        self.joint_names = []
        self._chain = []
        limits = []
        for joint in chain:
            kind = joint.get('type')
            axis = None
            if kind == 'revolute':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                axis /= np.linalg.norm(axis)
                limit = joint.find('limit')
                limits.append([float(limit.get(name)) for name in ('lower', 'upper', 'velocity')])
                self.joint_names.append(joint.get('name'))
                self.tool_parent_link = joint.find('child').get('link')
                self.parent_to_tool = np.eye(4)
            elif kind != 'fixed':
                raise ValueError(f'机械臂运动链不支持 {kind} 关节')
            origin = _origin(joint)
            self._chain.append((origin, axis))
            if kind == 'fixed' and limits:
                self.parent_to_tool = self.parent_to_tool @ origin
        if len(limits) != 7:
            raise ValueError('机械臂运动链必须包含七个转动关节')
        self.lower, self.upper, self.velocity_limits = np.array(limits).T

    def forward(self, q):
        """将七个关节角转换成 base_link_T_tcp，包含底座和末端固定变换。"""
        pose = np.eye(4)
        index = 0
        for origin, axis in self._chain:
            pose = pose @ origin
            if axis is not None:
                pose[:3, :3] = (
                    pose[:3, :3] @ Rotation.from_rotvec(axis * q[index]).as_matrix()
                )
                index += 1
        return pose

    def _error(self, q, target):
        current = self.forward(q)
        return (
            current[:3, 3] - target[:3, 3],
            Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
        )

    def solve(self, target, seed):
        """在关节限位内求解；独立验收 1 mm / 0.01 rad，不把优化器退出当作到位。"""
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (7,) or not np.all(np.isfinite(seed)):
            raise ValueError('IK 初始角度必须是七个有限数值')
        if np.any(seed < self.lower) or np.any(seed > self.upper):
            raise ValueError('IK 初始角度超出 URDF 关节限位')
        target = np.asarray(target, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError('IK 目标必须是有限的 4×4 位姿矩阵')

        def residual(q):
            position, angle = self._error(q, target)
            # 位姿优先；很小的姿态偏好只用于约束冗余自由度，保持连续。
            return np.concatenate((
                position / 0.001,
                angle / 0.01,
                0.001 * (q - seed) / (self.upper - self.lower),
            ))

        errors = []
        for start in (seed, (self.lower + self.upper) / 2):
            result = least_squares(
                residual, start, bounds=(self.lower, self.upper),
                x_scale=self.upper - self.lower,
                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300,
            )
            position, angle = self._error(result.x, target)
            error = (float(np.linalg.norm(position)), float(np.linalg.norm(angle)))
            if error[0] <= 0.001 and error[1] <= 0.01:
                return result.x
            errors.append(error)
        position, angle = min(errors, key=lambda e: max(e[0] / 0.001, e[1] / 0.01))
        raise ValueError(f'IK 不可达或未收敛：位置残差 {position:.6f} m，姿态残差 {angle:.6f} rad')
