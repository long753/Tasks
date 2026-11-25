"""复用定位与规划入口，启动 DWB 控制器并执行一次导航任务。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("navigation_pkg"))
    controller = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )
    mission = Node(
        package="navigation_pkg",
        executable="navigation_task_node",
        name="navigation_task_node",
        parameters=[
            {
                "use_sim_time": True,
                "execute_path": True,
                "planner_id": ParameterValue(
                    LaunchConfiguration("planner_id"), value_type=str
                ),
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(share / "launch" / "planning.launch.py")
                ),
                launch_arguments={"standalone": "false"}.items(),
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=controller,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="Navigation controller exited"))
                    ],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=mission,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="Navigation mission exited"))
                    ],
                )
            ),
            controller,
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                parameters=[
                    {
                        "use_sim_time": True,
                        "autostart": True,
                        "node_names": [
                            "map_server",
                            "amcl",
                            "planner_server",
                            "controller_server",
                        ],
                    }
                ],
                output="screen",
            ),
            mission,
        ]
    )
