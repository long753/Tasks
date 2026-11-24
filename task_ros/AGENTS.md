# 项目说明

- `task_ros` 是 ROS 2 Humble 与 Gazebo Fortress 的移动操作仿真练习：差速底盘绕行至作业台，七轴 Franka Panda 与双指夹爪负责抓放。仓库提供场景、机器人、控制接口、已知地图 AMCL 定位及 A* 全局路径规划，不提供完整的局部避障、抓放或作业规划算法。
- 起点为 `(-3, 2)`，停靠点为 `(3, -1.3)`，朝南作业。桌台中心为 `(3, -2.2)`，顺时针旋转 90°；托盘和物块靠近桌面北侧，桌台西侧留出绕行空间。
- `asset/workcell.sdf` 定义静态场地和作业区；`asset/moving_obstacle.sdf` 定义可选的滑轨移动障碍；`asset/mobile_manipulator.urdf.xacro` 定义底盘、激光雷达和 Panda 的安装与关节接口。
- `scripts/world.launch.py` 启动仿真和控制器。移动障碍默认停用；`moving_obstacle:=true` 同时生成模型、启用 `config/moving_obstacle_bridge.yaml` 桥接并运行 `scripts/moving_obstacle.py`，不再通过注释模型切换实验。
- `config/bridge.yaml` 配置基础 ROS/Gazebo 话题桥接，`config/controllers.yaml` 配置关节状态、机械臂轨迹和夹爪力控。
- `src/auto_pkg/` 只放算法练习节点。现有 `navigation` 是前方测距停车示例，不是完整导航；尚未实现的抓放节点不注册入口。

## 运行环境

使用 Ubuntu 22.04 的系统 Python 3.10 和 APT 安装的 ROS 2 依赖，不使用 uv。从 `task_ros` 目录运行：

```bash
source /opt/ros/humble/setup.bash
ros2 launch scripts/world.launch.py
```

仿真不显示 Gazebo 窗口时加 `gui:=false`；启用移动障碍时运行 `ros2 launch scripts/world.launch.py moving_obstacle:=true`。

### Gazebo 真值激光建图

- 使用系统 `python3-numpy`、`python3-pil` 和 `ros-humble-rviz2`。先启动仿真，在已加载 ROS 环境的另一终端运行 `ros2 launch scripts/mapping.launch.py`；launch 负责建图进程和 RViz，RViz 显示 `/map` 顶视图。
- 不需要 RViz 时运行 `ros2 launch scripts/mapping.launch.py rviz:=false`，也可直接运行 `python3 scripts/mapping.py`。建图脚本不再自行启动或管理 RViz，也不再使用 `--no-rviz` 参数。
- 建图脚本按仿真时间对齐 `/scan` 和 `/ground_truth/odom`，用 `base_link → laser_link` 安装变换投影激光。它不做扫描匹配，不使用轮式 `/odom`，也不发送 `/cmd_vel`。在独立终端驾驶；雷达没扫到的区域保留为未知，移动障碍可能留下痕迹。
- 建图完成后先停稳，在建图终端按 `Ctrl+C` 保存 `outputs/map/map.png` 和 `outputs/map/map.yaml`，等保存结束再关 Gazebo。只关 RViz 不会保存。
- 栅格分辨率为 0.05 m，`map` 的 XY 与 Gazebo `world` 对齐；范围直接读取当前轴对齐场地的 SDF 地板尺寸，并包含边墙，不另维护一份尺寸配置。后续导航加载 YAML（PNG 与 YAML 同目录）；建图脚本不发布 `map → odom`，也不承担导航定位。
- `scripts/mapping.py` 中 `WorkcellGrid` 只处理 NumPy 栅格，按激光投影、射线离散化、证据更新分步组织；`GroundTruthMapping` 只处理 ROS 消息、时间同步和发布。命中证据 +3、空闲 -1，限制在 [-6, 6]；证据 >=2 为占用，<0 为空闲，其余未知。
- 地图直接写入 PNG 和 YAML，不使用临时文件替换；保存时若中断或写盘失败，需重新保存。未收到同步扫描时不保存；意外错误保留 traceback。

### 已知地图 AMCL 定位（导航第一阶段）

- 依赖 APT 包 `ros-humble-nav2-map-server`、`ros-humble-nav2-amcl`、`ros-humble-nav2-lifecycle-manager` 和 `ros-humble-rviz2`，使用系统 ROS 2 Humble 环境。
- 先运行 `ros2 launch scripts/world.launch.py`，保持移动障碍停用。在另一终端加载 ROS 环境后运行 `ros2 launch scripts/localization.launch.py`；不要同时运行建图程序或旧 `navigation` 测距停车示例。
- 定位入口只启动 `map_server`、`amcl`、定位生命周期管理器和可选 RViz，不启动规划器、代价地图、局部控制器，也不发布 `/cmd_vel`。参数在 `config/localization.yaml`，显示配置在 `config/localization.rviz`。
- 默认加载 `outputs/map/map.yaml`；指定其他地图用 `map:=/绝对路径/map.yaml`，关闭 RViz 用 `rviz:=false`。地图图像仍按 YAML 中的相对路径加载，定位不会改写地图。
- RViz 使用 `map` 固定坐标系，显示地图、激光、AMCL 位姿和协方差。AMCL 启动时自动读取 `asset/workcell.sdf` 中 `start_marker` 的 XY 和 yaw，默认是 `(-3, 2, 0)`，不再需要手动点击 **2D Pose Estimate**。`world.launch.py` 从同一标记生成机器人，修改起点不必同步两份坐标；收到有效扫描后自动建立定位链路。
- 使用已有 `base_link`，TF 链为 `map → odom → base_link → laser_link`：AMCL 只发布 `map → odom`，Gazebo DiffDrive 提供轮式里程计和 `odom → base_link`，`robot_state_publisher` 提供安装变换。全部节点使用仿真时间，不增加 `base_footprint` 或静态 `map → odom`。
- AMCL 初始化后仍只使用地图、激光和轮式里程计 TF，不订阅 `/ground_truth/odom`，也不把位姿持续锁死在出生点或真值上。真值只用于独立验证；可在另一终端低速手动驾驶，观察激光与墙体重合及位姿是否稳定，不代跑导航算法。
- `/particle_cloud` 为 `nav2_msgs/msg/ParticleCloud`；本 RViz 配置使用原生位姿与协方差显示，不依赖额外的 Nav2 RViz 插件，也不将粒子云错误配置为 `PoseArray`。
- 自动初始化假设定位启动时机器人仍在场景起点，且地图与 Gazebo `world` 对齐。机器人已离开起点时，单独重启定位仍会使用出生位姿；应先重启场景回到起点，或使用保留的 **2D Pose Estimate** 工具重新定位。

### A* 全局路径规划（导航第二阶段）

- 在定位依赖之外安装 `ros-humble-nav2-planner` 和 `ros-humble-nav2-navfn-planner`，APT 会同时安装代价地图依赖；使用系统 Python 的 NumPy，无需构建额外 ROS 包。
- 先启动 `ros2 launch scripts/world.launch.py`，保持移动障碍停用。在另一终端加载 ROS 环境后运行 `ros2 launch scripts/planning.launch.py`；此入口已包含定位，不要再单独启动 `localization.launch.py`、建图或旧 `navigation` 示例。支持 `map:=/绝对路径/map.yaml` 和 `rviz:=false`。
- `scripts/planning_map.py` 读取场景工作台桌面尺寸和位置，在 `/map` 的副本中把桌面投影标为占用，发布 `/planning_map`。原始 `/map` 仍只供 AMCL 使用，规划节点不改写已有 PNG/YAML。投影会组合模型、链接和碰撞体的平面 yaw；当前 90° 转向得到精确矩形，其他平面角度使用保守的轴对齐包围盒。
- `config/planning.yaml` 使用 Nav2 的 `nav2_navfn_planner/NavfnPlanner`，明确设置 `use_astar: true`、`allow_unknown: false`、`tolerance: 0.0`。全局代价地图只使用静态层和膨胀层，基于 `map`、`base_link` 和仿真时间，不接入动态障碍层。
- 当前收拢姿态的碰撞几何最大 XY 投影半径约为 0.442 m，建议采用含余量的 0.50 m 圆形轮廓。`inflation_radius` 不应小于 `robot_radius`，否则膨胀层可能漏掉碰撞边界；具体参数在 `config/planning.yaml` 中调整。规划时保持机械臂收拢，不能把这个固定圆形轮廓用于伸臂状态。
- `scripts/plan_goal.py` 启动时从场景 `dock_marker` 读取目标 XY 和 yaw，等待 AMCL 提供 `map → base_link` 变换且规划器生命周期进入 active，再自动请求一次 `/compute_path_to_pose`。起点来自当前定位 TF，不需要手动设置目标位置或朝向。
- 当前停靠目标为 `(3, -1.3)`、朝向南侧，距桌面北边 0.60 m，保留原有 0.50 m 规划半径。目标在障碍、未知区或地图外仍会明确失败，不自动替换成附近目标。桌台调整后已用实际激光更新受影响地图区域，区域外观测与 YAML 参数保持不变。
- RViz 显示 **Docking Goal** 和 **Docking Global Plan**，分别来自 `/goal_pose` 与 `/global_plan`；两者保留最后一条消息，支持 RViz 晚启动。规划失败会明确记录，不自动重试或替换目标；修改 `dock_marker` 后须重新启动仿真与规划，保证目标与实际场景一致。
- 第二阶段只计算并显示路径，不启动局部控制器、行为树导航或路径跟踪，不发布 `/cmd_vel`，机器人不会自动行驶。

## 维护约定

- `README.md` 由用户维护。不要修改本项目或上级仓库的 `README.md`；项目说明和面向代理的约定写在本文件。
- 只有用户明确要求时才创建 Git 提交或推送；完成代码修改后默认保留为未提交改动，不自动提交。
- 修改仿真环境时只验证受影响的模型和接口；未经用户要求，不实现或代跑避障、抓放和作业规划练习。
- 保持 `asset/` 模型、`config/` 配置、`scripts/` 仿真与建图入口、`src/auto_pkg/` 算法练习的职责划分；`build/`、`install/`、`log/` 和 `__pycache__/` 是生成物，不是源码。
- 可读性优先于通用化：不预留空节点，不引入工具包、基类或配置管理框架。保留激光有效值与越界处理、按时间戳查询真值 TF、仿真时间和控制器启动依赖，这些属于正确性约束。