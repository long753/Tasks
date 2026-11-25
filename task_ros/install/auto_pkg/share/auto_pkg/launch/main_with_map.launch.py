"""Start the workcell, navigate to its dock, then pick and place the target."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    auto_share = Path(get_package_share_directory("auto_pkg"))
    navigation_share = Path(get_package_share_directory("navigation_pkg"))
    manipulation_share = Path(get_package_share_directory("manipulation_pkg"))

    def start_navigation(event, _context):
        if event.returncode != 0:
            return []
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(
                        navigation_share
                        / "launch"
                        / "navigation_with_map.launch.py"
                    )
                ),
                launch_arguments={
                    name: LaunchConfiguration(name)
                    for name in ("map", "params_file", "planner_id", "rviz")
                }.items(),
            ),
            GroupAction(actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        str(manipulation_share / "launch" / "manipulation.launch.py")
                    ),
                    launch_arguments={
                        "wait_for_navigation": "true",
                        "params_file": LaunchConfiguration("manipulation_params_file"),
                    }.items(),
                ),
            ]),
        ]

    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(auto_share / "launch" / "world.launch.py")),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "moving_obstacle": LaunchConfiguration("moving_obstacle"),
        }.items(),
    )
    navigation_after_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=lambda action: (
                isinstance(action, Node)
                and action.node_name == "/robot_controller_spawner"
            ),
            on_exit=start_navigation,
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui", default_value="true", choices=["true", "false"]
            ),
            DeclareLaunchArgument(
                "moving_obstacle",
                default_value="false",
                choices=["true", "false"],
                description="同时启用移动障碍模型、桥接和驱动",
            ),
            DeclareLaunchArgument(
                "map", default_value=str(navigation_share / "maps" / "map.yaml")
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(navigation_share / "config" / "nav2.yaml"),
            ),
            DeclareLaunchArgument(
                "manipulation_params_file",
                default_value=str(manipulation_share / "config" / "manipulation.yaml"),
            ),
            DeclareLaunchArgument("planner_id", default_value="GridBased"),
            DeclareLaunchArgument(
                "rviz", default_value="true", choices=["true", "false"]
            ),
            navigation_after_controllers,
            world,
        ]
    )
