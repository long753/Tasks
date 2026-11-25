"""启动已知地图定位、A* 规划和可选的单次规划任务，不驱动机器人。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from navigation_pkg.scene import marker_pose


def generate_launch_description():
    share = Path(get_package_share_directory("navigation_pkg"))
    params_file = LaunchConfiguration("params_file")
    start = marker_pose("start_marker")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map", default_value=str(share / "maps" / "map.yaml")
            ),
            DeclareLaunchArgument(
                "params_file", default_value=str(share / "config" / "nav2.yaml")
            ),
            DeclareLaunchArgument("planner_id", default_value="GridBased"),
            DeclareLaunchArgument(
                "rviz", default_value="true", choices=["true", "false"]
            ),
            DeclareLaunchArgument(
                "standalone",
                default_value="true",
                choices=["true", "false"],
                description="独立运行时管理生命周期并请求路径；导航入口关闭后统一管理",
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                parameters=[
                    params_file,
                    {
                        "yaml_filename": ParameterValue(
                            LaunchConfiguration("map"), value_type=str
                        ),
                    },
                ],
                output="screen",
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                parameters=[
                    params_file,
                    {
                        "initial_pose.x": start[0],
                        "initial_pose.y": start[1],
                        "initial_pose.z": 0.0,
                        "initial_pose.yaw": start[5],
                    },
                ],
                output="screen",
            ),
            Node(
                package="navigation_pkg",
                executable="planning_map_node",
                name="planning_map_node",
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                parameters=[params_file],
                output="screen",
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_planning",
                condition=IfCondition(LaunchConfiguration("standalone")),
                parameters=[
                    {
                        "use_sim_time": True,
                        "autostart": True,
                        "node_names": ["map_server", "amcl", "planner_server"],
                    }
                ],
                output="screen",
            ),
            Node(
                package="navigation_pkg",
                executable="navigation_task_node",
                name="navigation_task_node",
                condition=IfCondition(LaunchConfiguration("standalone")),
                parameters=[
                    {
                        "use_sim_time": True,
                        "execute_path": False,
                        "planner_id": ParameterValue(
                            LaunchConfiguration("planner_id"), value_type=str
                        ),
                    }
                ],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(share / "config" / "navigation.rviz")],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ]
    )
