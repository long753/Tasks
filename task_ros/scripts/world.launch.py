"""Start the Fortress workcell, robot, ROS 2 bridge and controllers."""

import os
import sys
from pathlib import Path
from xml.etree import ElementTree

import xacro
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    GroupAction,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
ASSET = PROJECT / "asset"
CONFIG = PROJECT / "config"


def start_workcell(context):
    # 场景起点同时作为机器人出生位置和 AMCL 的初始估计。
    world = ElementTree.parse(ASSET / "workcell.sdf")
    start_pose = tuple(
        map(float, world.findtext("./world/model[@name='start_marker']/pose").split())
    )
    # 1. 展开机器人描述，并补充 Gazebo 的插件与资源搜索路径。
    description = xacro.process_file(
        str(ASSET / "mobile_manipulator.urdf.xacro"),
        mappings={"controllers_file": str(CONFIG / "controllers.yaml")},
    ).toxml()
    mode = [] if LaunchConfiguration("gui").perform(context) == "true" else ["-s"]
    plugin_paths = [
        str(Path(get_package_prefix("gz_ros2_control")) / "lib"),
        os.environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", ""),
    ]
    resource_paths = [
        str(Path(get_package_prefix("franka_description")) / "share"),
        os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
    ]
    # 2. 启动仿真、机器人状态发布和基础话题桥接。
    simulator = ExecuteProcess(
        cmd=[
            "ign",
            "gazebo",
            "-r",
            "-v",
            "3",
            *mode,
            str(ASSET / "workcell.sdf"),
        ],
        name="workcell_simulator",
        output="screen",
        additional_env={
            "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": os.pathsep.join(
                filter(None, plugin_paths)
            ),
            "IGN_GAZEBO_RESOURCE_PATH": os.pathsep.join(filter(None, resource_paths)),
        },
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": ParameterValue(description, value_type=str),
                "use_sim_time": True,
            }
        ],
        output="screen",
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="workcell_bridge",
        parameters=[{"config_file": str(CONFIG / "bridge.yaml")}],
        output="screen",
    )
    # 3. 生成机器人；成功后再加载控制器，不能用固定延时替代此依赖。
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-world",
            "workcell",
            "-topic",
            "/robot_description",
            "-name",
            "mobile_manipulator",
            "-x",
            str(start_pose[0]),
            "-y",
            str(start_pose[1]),
            "-z",
            "0.101",
            "-Y",
            str(start_pose[5]),
        ],
    )
    controllers = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "arm_controller",
            "gripper_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "45",
        ],
    )
    # 4. 一个开关同时控制移动障碍模型、话题桥接和运动驱动。
    obstacle = GroupAction(
        condition=IfCondition(LaunchConfiguration("moving_obstacle")),
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                name="spawn_moving_obstacle",
                arguments=[
                    "-world", "workcell",
                    "-file", str(ASSET / "moving_obstacle.sdf"),
                    "-name", "moving_obstacle",
                    "-z", "0.3",
                ],
                output="screen",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="moving_obstacle_bridge",
                parameters=[
                    {"config_file": str(CONFIG / "moving_obstacle_bridge.yaml")}
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[sys.executable, str(SCRIPTS / "moving_obstacle.py")],
                name="moving_obstacle_driver",
                output="screen",
            ),
        ],
    )

    def after_spawn(event, _context):
        if event.returncode == 0:
            return [controllers]
        return [EmitEvent(event=Shutdown(reason="Failed to spawn the workcell robot"))]

    def after_controller_setup(event, _context):
        if event.returncode != 0:
            return [
                EmitEvent(
                    event=Shutdown(reason="Failed to start the workcell controllers")
                )
            ]
        return []

    return [
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_robot, on_exit=after_spawn)
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=controllers,
                on_exit=after_controller_setup,
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=simulator,
                on_exit=[EmitEvent(event=Shutdown(reason="Gazebo exited"))],
            )
        ),
        simulator,
        state_publisher,
        bridge,
        spawn_robot,
        obstacle,
    ]


def generate_launch_description():
    return LaunchDescription(
        initial_entities=[
            DeclareLaunchArgument(
                name="gui", default_value="true", choices=["true", "false"]
            ),
            DeclareLaunchArgument(
                name="moving_obstacle", default_value="false", choices=["true", "false"],
                description="同时启用移动障碍模型、桥接和驱动",
            ),
            OpaqueFunction(function=start_workcell),
        ]
    )
