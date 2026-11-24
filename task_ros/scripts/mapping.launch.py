"""Run ground-truth mapping and optionally display /map in RViz."""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description():
    mapper = ExecuteProcess(
        cmd=[sys.executable, str(ROOT / "scripts" / "mapping.py")],
        name="ground_truth_mapping",
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", str(ROOT / "config" / "mapping.rviz")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz", default_value="true", choices=["true", "false"],
                description="显示 /map 顶视图；关闭 RViz 不会停止建图",
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=mapper,
                    on_exit=[EmitEvent(event=Shutdown(reason="Mapping node exited"))],
                )
            ),
            mapper,
            rviz,
        ]
    )
