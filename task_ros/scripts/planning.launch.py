"""Run AMCL and automatically plan to the workcell dock without driving."""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
SCRIPTS = ROOT / "scripts"


def generate_launch_description():
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(SCRIPTS / "localization.launch.py")),
        launch_arguments={
            "map": LaunchConfiguration("map"),
            "rviz": "false",
        }.items(),
    )
    planning_map = ExecuteProcess(
        cmd=[sys.executable, str(SCRIPTS / "planning_map.py")],
        name="planning_map",
        output="screen",
    )
    planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        parameters=[str(CONFIG / "planning.yaml")],
        output="screen",
    )
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_planning",
        parameters=[
            {
                "use_sim_time": True,
                "autostart": True,
                "node_names": ["planner_server"],
            }
        ],
        output="screen",
    )
    goal_planner = ExecuteProcess(
        cmd=[sys.executable, str(SCRIPTS / "plan_goal.py")],
        name="plan_goal",
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", str(CONFIG / "planning.rviz")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=str(ROOT / "outputs" / "map" / "map.yaml"),
                description="Saved map YAML passed to AMCL and map_server",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                choices=["true", "false"],
                description="Display planning layers and the global path",
            ),
            # 子入口的 rviz=false 只在组内生效，不覆盖规划界面的 rviz 参数。
            GroupAction(actions=[localization]),
            planning_map,
            planner,
            lifecycle_manager,
            goal_planner,
            rviz,
        ]
    )
