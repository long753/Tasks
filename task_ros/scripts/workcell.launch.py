"""Start the Fortress workcell, robot, ROS 2 bridge and controllers."""

import os
import sys
from pathlib import Path
from shlex import quote

import xacro
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
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
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError(
            "Use bash scripts/ros_uv.sh ros2 launch scripts/workcell.launch.py"
        )
    description = xacro.process_file(
        str(ASSET / "robots" / "mobile_manipulator.urdf.xacro"),
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
    simulator = ExecuteProcess(
        cmd=[
            "ign",
            "gazebo",
            "-r",
            "-v",
            "3",
            *mode,
            str(ASSET / "worlds" / "workcell.sdf"),
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
            "-3.0",
            "-y",
            "0.0",
            "-z",
            "0.101",
        ],
    )
    controllers = Node(
        package="controller_manager",
        executable="spawner",
        prefix=[quote(sys.executable)],
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
    obstacle = ExecuteProcess(
        cmd=[sys.executable, str(SCRIPTS / "moving_obstacle.py")],
        name="moving_obstacle_driver",
        output="screen",
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
        [
            DeclareLaunchArgument(
                "gui", default_value="true", choices=["true", "false"]
            ),
            OpaqueFunction(function=start_workcell),
        ]
    )
