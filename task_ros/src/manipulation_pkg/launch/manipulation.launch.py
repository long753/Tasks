"""Launch only the pick_place_node manipulation node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("manipulation_pkg"))

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "wait_for_navigation", default_value="false", choices=["true", "false"]
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(share / "config" / "manipulation.yaml"),
            ),
            DeclareLaunchArgument(
                "scene_file",
                default_value=str(share / "asset" / "workcell.sdf"),
            ),
            Node(
                package="manipulation_pkg",
                executable="pick_place_node",
                name="pick_place_node",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "scene_file": ParameterValue(
                            LaunchConfiguration("scene_file"), value_type=str
                        ),
                        "wait_for_navigation": ParameterValue(
                            LaunchConfiguration("wait_for_navigation"), value_type=bool
                        ),
                    },
                ],
                output="screen",
            ),
        ]
    )
