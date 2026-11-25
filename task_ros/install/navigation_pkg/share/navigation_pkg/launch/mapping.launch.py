"""Run ground-truth-assisted mapping without starting the simulator."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mapper = Node(
        package="navigation_pkg",
        executable="mapping_node",
        name="mapping_node",
        parameters=[
            {
                "use_sim_time": True,
                "output_dir": ParameterValue(
                    LaunchConfiguration("output_dir"), value_type=str
                ),
            }
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output_dir",
                default_value=str(Path.cwd() / "outputs" / "map"),
                description="Directory where map.yaml and map.png are saved",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                choices=["true", "false"],
                description="Display the raw /map topic in RViz",
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=mapper,
                    on_exit=[EmitEvent(event=Shutdown(reason="Mapping node exited"))],
                )
            ),
            mapper,
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=[
                    "-d",
                    str(
                        Path(get_package_share_directory("navigation_pkg"))
                        / "config"
                        / "mapping.rviz"
                    ),
                ],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ]
    )
