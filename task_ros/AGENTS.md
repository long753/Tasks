# 项目说明

- `task_ros` 是 ROS 2 Humble 与 Gazebo Fortress 的移动操作仿真练习：差速底盘绕行至作业台，七轴 Franka Panda 与双指夹爪负责抓放。仓库目前提供场景、机器人和控制接口，不提供自主避障、抓放或作业规划算法。
- 起点 `(-3, 2)` 与停靠点 `(1.82, -2)` 位于场地对角，作业台及托盘与停靠点对齐。南北保留绕行空间，移动障碍沿原滑轨运行。
- `asset/workcell.sdf` 定义场地、起点与停靠标记、作业台、托盘、物块和障碍；`asset/mobile_manipulator.urdf.xacro` 定义底盘、激光雷达和 Panda 的安装与关节接口。
- `scripts/main.launch.py` 启动仿真；`scripts/moving_obstacle.py` 按仿真时钟驱动移动障碍。`config/bridge.yaml` 配置 ROS 与 Gazebo 的话题桥接，`config/controllers.yaml` 配置关节状态、机械臂轨迹和夹爪力控。

## 运行环境

使用 Ubuntu 22.04 的系统 Python 3.10 和 APT 安装的 ROS 2 依赖，不使用 uv。从 `task_ros` 目录运行：

```bash
source /opt/ros/humble/setup.bash
ros2 launch scripts/main.launch.py
```

### Gazebo 真值激光建图

- 使用系统 `python3-numpy`、`python3-pil` 和 `ros-humble-rviz2`。用 `scripts/main.launch.py` 启动仿真，在已加载 ROS 环境的另一终端执行 `python3 scripts/mapping.py`；RViz 会显示 `/map` 顶视图。无桌面时加 `--no-rviz`。
- 建图脚本按仿真时间对齐 `/scan` 和 `/ground_truth/odom`，用 `base_link → laser_link` 安装变换投影激光。它不做扫描匹配，不使用轮式 `/odom`，也不发送 `/cmd_vel`。在独立终端驾驶；雷达没扫到的区域保留为未知，移动障碍可能留下痕迹。
- 建图完成后先停稳，在建图终端按 `Ctrl+C` 保存 `outputs/map/map.png` 和 `outputs/map/map.yaml`，等保存结束再关 Gazebo。只关 RViz 不会保存。
- 栅格分辨率为 0.05 m，`map` 的 XY 与 Gazebo `world` 对齐，覆盖场地地板和边墙。后续导航加载 YAML（PNG 与 YAML 同目录）；建图脚本不发布 `map → odom`，也不承担导航定位。

## 维护约定

- `README.md` 由用户维护。不要修改本项目或上级仓库的 `README.md`；项目说明和面向代理的约定写在本文件。
- 修改仿真环境时只验证受影响的模型和接口；未经用户要求，不实现或代跑避障、抓放和作业规划练习。