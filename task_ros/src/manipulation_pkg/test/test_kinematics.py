"""Pure kinematics checks; no ROS, Gazebo, filesystem fixtures or running services."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manipulation_pkg.kinematics import ArmKinematics


def arm():
    joints = []
    for i in range(7):
        parent = 'mount' if i == 0 else f'link{i - 1}'
        joints.append(f'''
          <joint name="joint{i}" type="revolute">
            <parent link="{parent}"/><child link="link{i}"/>
            <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
            <limit lower="-1" upper="1" velocity="0.5" effort="10"/>
          </joint>''')
    return ArmKinematics('''<robot name="kinematics_fixture">
        <joint name="mount" type="fixed">
          <parent link="base_link"/><child link="mount"/>
          <origin xyz="0.2 -0.1 0.3" rpy="0 0 0.4"/>
        </joint>''' + ''.join(joints) + '''
        <joint name="tcp" type="fixed">
          <parent link="link6"/><child link="fer_hand_tcp"/>
          <origin xyz="0.1 0 0.05" rpy="3.141592653589793 0 0"/>
        </joint></robot>''')


def test_forward_includes_mount_and_tcp_offsets():
    model = arm()
    pose = model.forward(np.zeros(7))
    np.testing.assert_allclose(
        pose[:3, 3], [0.2 + 0.1 * np.cos(0.4), -0.1 + 0.1 * np.sin(0.4), 1.05],
        atol=1e-12,
    )
    expected_rotation = Rotation.from_euler('z', 0.4).as_matrix() @ np.diag([1., -1., -1.])
    np.testing.assert_allclose(pose[:3, :3], expected_rotation, atol=1e-12)
    np.testing.assert_allclose(model.parent_to_tool[:3, 3], [0.1, 0., 0.05], atol=1e-12)


def test_inverse_handles_near_pi_rotation_with_joint_limits():
    model = arm()
    angle = np.pi - 0.01
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler('z', 0.4 + angle).as_matrix() @ np.diag([1., -1., -1.])
    target[:3, 3] = [0.2 + 0.1 * np.cos(0.4 + angle), -0.1 + 0.1 * np.sin(0.4 + angle), 1.05]
    result = model.solve(target, np.zeros(7))
    actual = model.forward(result)
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) <= 0.001
    assert Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude() <= 0.01
    assert np.all(result >= model.lower) and np.all(result <= model.upper)


def test_unreachable_pose_fails_instead_of_returning_an_approximate_grasp():
    model = arm()
    target = np.eye(4)
    target[:3, 3] = [10., 0., 0.]
    with pytest.raises(ValueError, match='残差'):
        model.solve(target, np.zeros(7))
