# 项目说明
- `task_ros` 是 ROS 2 Humble 与 Gazebo Fortress 的移动操作仿真练习：差速底盘绕行至作业台，七轴 Franka Panda 与双指夹爪负责抓放。仓库目前提供场景、机器人和控制接口，不提供自主避障、抓放或作业规划算法。
- 场地保留南北两侧的绕行空间；错位的静态挡板收窄通道，移动障碍仍沿原滑轨运行。起点、停靠点和作业台布局不变。
- `asset/workcell.sdf` 定义场地、起点与停靠标记、作业台、托盘、物块和障碍；`asset/mobile_manipulator.urdf.xacro` 定义底盘、激光雷达和 Panda 的安装与关节接口。
- `scripts/workcell.launch.py` 启动仿真；`scripts/moving_obstacle.py` 按仿真时钟驱动移动障碍。`config/bridge.yaml` 配置 ROS 与 Gazebo 的话题桥接，`config/controllers.yaml` 配置关节状态、机械臂轨迹和夹爪力控。

## 运行环境

使用 Ubuntu 22.04 的系统 Python 3.10 和 APT 安装的 ROS 2 依赖，不使用 uv。从 `task_ros` 目录运行：

```bash
source /opt/ros/humble/setup.bash
ros2 launch scripts/workcell.launch.py
```

## 维护约定

- `README.md` 由用户维护。不要修改本项目或上级仓库的 `README.md`；项目说明和面向代理的约定写在本文件。
- 修改仿真环境时只验证受影响的模型和接口；未经用户要求，不实现或代跑避障、抓放和作业规划练习。