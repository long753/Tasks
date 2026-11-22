# Learning Tasks

This repository brings together my learning tasks and experiments including:

- [task_api](task_api/): Detect and track objects in images and videos, then use a vision-language API to analyze their relationships.
- [task_cv](task_cv/): Compare object detection models on crack detection and explore Slot Attention for object-centric visual learning.
- [task_jepa](task_jepa/): Train and evaluate H-JEPA, HWM, and LeWM world models, with experiments on adaptive temporal hierarchies and hierarchical planning.
- [task_ros](#task_ros): Gazebo workcell for mobile obstacle avoidance and cube-to-tray manipulation with ROS 2.
- [task_siglip](task_siglip/): Train and evaluate SigLIP models for image–text alignment and retrieval.
- [task_smolval](task_smolval/): Fine-tune SmolVLA for a Panda robot's cube-to-tray task in MuJoCo and compare performance before and after fine-tuning.

Each task has its own environment, dependencies, and run instructions; see its README before installing or running it. 

## `task_ros`

Use Ubuntu 22.04, ROS 2 Humble, and Gazebo Fortress. With Humble Desktop installed, the workcell also needs:

```bash
sudo apt install ignition-fortress ros-humble-ros-gz ros-humble-xacro \
  ros-humble-gz-ros2-control ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller ros-humble-forward-command-controller \
  ros-humble-franka-description
```

From the repository root, start the world, spawn the robot, and bring up the ROS 2 bridge and controllers:

```bash
cd task_ros
source /opt/ros/humble/setup.bash
ros2 launch asset/launch/workcell.launch.py
```

Append `gui:=false` to omit the desktop window; the GPU lidar still needs a rendering context. Stop with `Ctrl+C`.

The [world](task_ros/asset/worlds/workcell.sdf) has an 8 × 6 m floor, a blue start at `(-3, 0)`, and a green dock at `(1.82, 0)` facing world +X. A static obstacle leaves north and south routes. The red obstacle runs on a physical Y-axis slider at `x=0`, with a ±1.1 m target and a 12 s simulation-time period. The worktable is 0.72 m high; the yellow 50 mm block starts at `(2.43, -0.19, 0.745)` and the tray is centered at `(2.43, 0.19)`.

The [robot model](task_ros/asset/robots/mobile_manipulator.urdf.xacro) includes differential drive, a forward 2D lidar and the original seven-axis Franka Panda, named `fer` in the [official description package](https://frankarobotics.github.io/docs/doc/franka_description/README.html). It uses the stock Franka Hand with two simulated finger effort inputs. The [control configuration](task_ros/asset/config/controllers.yaml) exposes these ROS 2 interfaces:

| Interface | Use |
| --- | --- |
| `/cmd_vel` (`geometry_msgs/msg/Twist`) | Base command: `linear.x` in m/s, `angular.z` in rad/s. Send zero to stop. |
| `/scan`, `/odom`, `/tf`, `/tf_static`, `/clock` | Laser frame `laser_link`, wheel odometry `odom → base_link`, robot transforms and simulation time. Wheel odometry starts at the spawn pose, not the world origin. |
| `/joint_states` | Wheel, arm and gripper joint feedback. |
| `/arm_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` for `fer_joint1` through `fer_joint7`. |
| `/gripper_controller/commands` (`std_msgs/msg/Float64MultiArray`) | Effort in N, ordered `fer_finger_joint1`, `fer_finger_joint2`. Send equal forces; positive opens, negative closes. Effort remains commanded until replaced. |

The Panda tool frame is `fer_hand_tcp`, with +Z along the fingers and up to 80 mm nominal opening. The simulation adjusts only `fer_link4`'s inertia tensor from the official description to satisfy SDFormat's physical inertia check. The environment does not start Nav2, MoveIt 2, or a task planner.

### Edit the workcell

- **Layout:** edit `asset/worlds/workcell.sdf`. Move a whole model with its `<model>` pose; if you resize a shape, update both its collision and visual geometry. The blue start marker is visual only; to move the robot too, update `-x`, `-y`, and `-Y` in `asset/launch/workcell.launch.py`.
- **Moving obstacle:** edit the amplitude and period in `asset/launch/moving_obstacle.py`. Keep the target inside the slider joint limits in the SDF.
- **Robot:** edit the mount transform `panda_mount_joint` and the simulated starting joint values in `asset/robots/mobile_manipulator.urdf.xacro`, not the installed package. If you rename or reorder controlled joints, update `asset/config/controllers.yaml` and command publishers to match.
- **ROS topics:** edit `asset/config/bridge.yaml` when a Gazebo topic is renamed or its ROS counterpart should differ.

Validate a world edit with `ign sdf -k asset/worlds/workcell.sdf`, then relaunch and inspect clearance and contact in Gazebo. For a robot edit, expand the Xacro to URDF and run `check_urdf` on the result before relaunching.
