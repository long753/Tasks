"""Start the workcell simulation and its ROS interfaces."""

import os
from pathlib import Path
from xml.etree import ElementTree

import xacro
from ament_index_python.packages import get_package_prefix, get_package_share_directory
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


def generate_launch_description():
    def start_world(context):
        share = Path(get_package_share_directory("auto_pkg"))
        asset, config = share / "asset", share / "config"
        world_file = asset / "workcell.sdf"
        world = ElementTree.parse(world_file)
        start = tuple(
            map(
                float,
                world.findtext("./world/model[@name='start_marker']/pose").split(),
            )
        )
        description = xacro.process_file(
            str(asset / "mobile_manipulator.urdf.xacro"),
            mappings={"controllers_file": str(config / "ros2_control.yaml")},
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
            cmd=["ign", "gazebo", "-r", "-v", "3", *mode, str(world_file)],
            name="workcell_simulator",
            additional_env={
                "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": os.pathsep.join(
                    filter(None, plugin_paths)
                ),
                "IGN_GAZEBO_RESOURCE_PATH": os.pathsep.join(
                    filter(None, resource_paths)
                ),
            },
            output="screen",
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
            parameters=[{"config_file": str(config / "bridge.yaml")}],
            output="screen",
        )
        spawn_robot = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-world",
                "workcell",
                "-topic",
                "/robot_description",
                "-name",
                "mobile_manipulator",
                "-x",
                str(start[0]),
                "-y",
                str(start[1]),
                "-z",
                "0.101",
                "-Y",
                str(start[5]),
            ],
            output="screen",
        )
        controllers = Node(
            package="controller_manager",
            executable="spawner",
            name="robot_controller_spawner",
            namespace="/",
            arguments=[
                "joint_state_broadcaster",
                "arm_controller",
                "gripper_controller",
                "--controller-manager",
                "/controller_manager",
                "--controller-manager-timeout",
                "45",
            ],
            output="screen",
        )
        obstacle = GroupAction(
            condition=IfCondition(LaunchConfiguration("moving_obstacle")),
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    name="spawn_moving_obstacle",
                    arguments=[
                        "-world",
                        "workcell",
                        "-file",
                        str(asset / "moving_obstacle.sdf"),
                        "-name",
                        "moving_obstacle",
                        "-z",
                        "0.3",
                    ],
                    output="screen",
                ),
                Node(
                    package="ros_gz_bridge",
                    executable="parameter_bridge",
                    name="moving_obstacle_bridge",
                    parameters=[
                        {"config_file": str(config / "moving_obstacle_bridge.yaml")}
                    ],
                    output="screen",
                ),
                Node(
                    package="auto_pkg",
                    executable="moving_obstacle_node",
                    output="screen",
                ),
            ],
        )

        def after_spawn(event, _context):
            if event.returncode == 0:
                return [controllers]
            return [EmitEvent(event=Shutdown(reason="Failed to spawn the workcell robot"))]

        def after_controllers(event, _context):
            if event.returncode != 0:
                return [
                    EmitEvent(event=Shutdown(reason="Failed to start robot controllers"))
                ]
            return []

        return [
            RegisterEventHandler(
                OnProcessExit(target_action=spawn_robot, on_exit=after_spawn)
            ),
            RegisterEventHandler(
                OnProcessExit(target_action=controllers, on_exit=after_controllers)
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
            OpaqueFunction(function=start_world),
        ]
    )
