"""Localize in a saved map with AMCL; no planner or velocity controller."""

from pathlib import Path
from xml.etree import ElementTree

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def generate_launch_description():
    # 与 world.launch.py 共用场景起点，避免另写一份出生坐标。
    world = ElementTree.parse(ROOT / "asset" / "workcell.sdf")
    start_pose = tuple(
        map(float, world.findtext("./world/model[@name='start_marker']/pose").split())
    )
    parameters = str(CONFIG / "localization.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=str(ROOT / "outputs" / "map" / "map.yaml"),
                description="已保存地图的 YAML 路径；图像与 YAML 放在同一目录",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                choices=["true", "false"],
                description="显示地图、激光、AMCL 位姿及其不确定性",
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                parameters=[
                    parameters,
                    {
                        "yaml_filename": ParameterValue(
                            LaunchConfiguration("map"), value_type=str
                        )
                    },
                ],
                output="screen",
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                parameters=[
                    parameters,
                    {
                        "initial_pose.x": start_pose[0],
                        "initial_pose.y": start_pose[1],
                        "initial_pose.z": 0.0,
                        "initial_pose.yaw": start_pose[5],
                    },
                ],
                output="screen",
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_localization",
                parameters=[
                    {
                        "use_sim_time": True,
                        "autostart": True,
                        "node_names": ["map_server", "amcl"],
                    }
                ],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(CONFIG / "localization.rviz")],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ]
    )
