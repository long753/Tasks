# 项目说明

`task_ros` 是 ROS 2 Humble、Gazebo Fortress 的移动操作仿真工程。当前实现仿真、真值辅助建图、AMCL、A*、Nav2 局部控制和真值 IK 抓放；不实现高层任务决策或无地图导航。

- 场景起点为 `(-3, 2)`；停靠点为 `(3, -1.3)`、朝南。桌台中心为 `(3, -2.2)`，顺时针转 90°，托盘和物块靠近桌面北侧。
- 根目录 `asset/` 是模型的唯一源码位置，保留 `workcell.sdf`、`moving_obstacle.sdf`、`mobile_manipulator.urdf.xacro`。包内资源链接用于安装到各包的 `share`，不要复制维护另一份模型。
- 根目录不再放 `scripts/` 或全局 `config/`。运行资源通过 `ament_index_python` 的包索引查找，不按源码父目录推算。

## 包职责

| 包 | 职责 |
|---|---|
| `auto_pkg` | 仿真、机器人控制接口、统一启动、可选移动障碍 |
| `navigation_pkg` | 建图、规划地图、AMCL、路径规划与单次导航任务 |
| `manipulation_pkg` | Gazebo 真值位姿、URDF 数值 IK、七轴轨迹、双指力控与六阶段抓放 |
| `planning_pkg` | 当前为空包，后续高层任务决策；不是全局路径规划包 |

- 自写 ROS 节点的文件、可执行入口和节点名均以 `_node` 结尾，文件头说明订阅、发布话题及动作接口。
- `auto_pkg/config/` 保存桥接和 `ros2_control.yaml`；`navigation_pkg/config/nav2.yaml` 按 Nav2 节点分区保存定位、规划、控制参数；`manipulation_pkg/config/manipulation.yaml` 保存抓取高度、倾角、速度与夹持力。省略已核实的默认项，不用 YAML 合并锚点或 Python 参数管理器。
- RViz 只保留 `navigation_pkg/config/mapping.rviz` 和 `navigation.rviz`。不为每个调试阶段复制一份配置。
- `auto_pkg` 依赖 `navigation_pkg` 和 `manipulation_pkg` 完成统一启动；算法包安装自己需要的场景资源，不反向依赖 `auto_pkg`，避免包依赖环。
- 空的 `planning_pkg` 不注册可执行程序；当前不创建 `main_without_map.launch.py`、`navigation_without_map.launch.py` 或其他占位节点。

## 构建与运行

使用 Ubuntu 22.04 的系统 Python 3.10 和 APT 安装的 ROS 2 依赖，不使用 uv。在工作空间根目录运行：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

每个新终端均需加载 ROS 和本工作空间的 `install/setup.bash`。依赖声明在各包 `package.xml`。

| 命令 | 行为 |
|---|---|
| `ros2 launch auto_pkg world.launch.py` | 只启动 Gazebo、机器人、桥接和关节控制器 |
| `ros2 launch auto_pkg main_with_map.launch.py` | 自行启动仿真，导航成功并停稳后抓放，最后机械臂复位；底盘不返航 |
| `ros2 launch navigation_pkg mapping.launch.py` | 在已运行的仿真中建图并显示 RViz |
| `ros2 launch navigation_pkg planning.launch.py` | 在已运行的仿真中启动定位并自动规划一次，不驱动机器人 |
| `ros2 launch navigation_pkg navigation_with_map.launch.py` | 在已运行的仿真中启动定位、规划并执行一次导航 |
| `ros2 launch manipulation_pkg manipulation.launch.py` | 在已停稳的工作台旁独立执行抓放；不要与完整主入口重复启动 |

- 同一时间只运行一套定位/导航入口；完整主入口不要与单独的 `world.launch.py` 同时运行。建图也不要与定位、规划或导航同时运行。
- 仿真入口支持 `gui:=false`；建图与导航入口支持 `rviz:=false`。
- `world.launch.py` 和 `main_with_map.launch.py` 都支持 `moving_obstacle:=true`，默认关闭；开启时同时生成模型、配置桥接并启动 `moving_obstacle_node`。导航期间机械臂收拢，停靠完成后再执行操作。本开关不改变现有一次性规划策略，也不代表动态障碍场景的完整导航抓放已经验收。
- 仿真启动动作直接在 `auto_pkg/launch/world.launch.py`，不再保留 `simulation.py` 或 `bringup.py` 启动辅助模块。先生成机器人，再加载关节控制器；`main_with_map.launch.py` 用 Include 组合入口，监听根命名空间下的 `/robot_controller_spawner` 成功退出后才启动导航，不用固定延时。
- 完整主入口用独立作用域包含操作 launch，分别用 `params_file` 和 `manipulation_params_file` 指定导航与操作参数，避免同名 launch 参数把 Nav2 文件误传给操作节点。
- 普通拷贝安装下，修改 launch/config 源码后需重新构建对应包再启动；不能只重启旧的 `install` 文件。本次统一入口开关已重新构建 `auto_pkg`，并实测障碍物实际运动。
- 启动文件只做编排，ROS 节点实现在各包的 `*_node.py` 中，通过 `console_scripts` 和 `launch_ros.actions.Node` 启动。

## 建图

- 本次按用户确认保留原有真值辅助建图，**不是 SLAM**，不做扫描匹配。`mapping_node` 订阅 `/scan`、`/ground_truth/odom` 和 `/tf_static`，只发布 `/map`，不发送速度。
- 使用扫描时间戳查询真值和 `base_link → laser_link` 安装变换。`WorkcellGrid` 只处理 NumPy 栅格，ROS 消息和时间同步由建图节点处理。
- 地图分辨率 0.05 m，`map` XY 与 Gazebo `world` 对齐。范围读取场景轴对齐地板并包含边墙；未观测区域保留未知。命中证据 +3、空闲 -1，限制在 [-6, 6]；证据 >=2 为占用、<0 为空闲，其余未知。
- 在独立终端驾驶，停稳后在建图终端按 `Ctrl+C` 保存。`output_dir` 默认是启动时工作目录下的 `outputs/map`，可传绝对路径；保存 PNG/YAML 后再关闭仿真，只关 RViz 不保存。
- 未收到同步扫描时不保存；直接写 PNG/YAML，不用临时文件替换。意外错误保留 traceback，写盘中断后需重新保存。

## 定位、规划与执行

- 默认地图来自已安装的 `navigation_pkg/maps/map.yaml`，源码链接到现有 `outputs/map`；已有 PNG/YAML 内容不因本次迁移改变。自定义地图用 `map:=/绝对路径/map.yaml`，图像仍按 YAML 相对路径加载。
- 地图必须与场景 `world` 对齐，才能直接使用 SDF 标记位姿。AMCL 仅在启动时读取 `start_marker` 初始化；机器人已离开起点时不要直接重启定位并假定仍在起点，可重启场景或用 RViz **2D Pose Estimate** 恢复。
- TF 为 `map → odom → base_link → laser_link`，AMCL 只发布 `map → odom`；Gazebo 提供轮式里程计 TF，状态发布器提供安装变换。导航不订阅真值里程计，也不持续锁定出生位姿。
- `planning_map_node` 在 `/map` 的副本上补充桌面占用并发布 `/planning_map`，不改写原始地图。组合模型、链接和碰撞体的平面 yaw，当前直角旋转为精确矩形，其他角度使用保守轴对齐包围盒。
- `planning.launch.py` 直接启动地图、AMCL、规划地图及规划器；独立运行时还启动生命周期管理器和单次规划任务。`navigation_with_map.launch.py` 通过 Include 设置 `standalone:=false` 复用公共节点，并自行启动控制器、执行任务和唯一的统一生命周期管理器，按地图、AMCL、规划器、控制器的顺序管理，避免重复任务和并行生命周期激活造成首次规划过早。
- 算法选择接口是 Nav2 插件配置与 `planner_id` 启动参数：在 `nav2.yaml` 的 `planner_server` 下配置插件 ID，再用 `planner_id:=对应ID` 选择；其他参数文件用 `params_file:=/绝对路径/nav2.yaml`。默认 `GridBased` 使用 NavFn A*，禁止未知区，目标容差为零，不替换失败目标。
- `navigation_task_node` 从 `dock_marker` 读取目标，等待定位及所需 Nav2 生命周期 active 后请求一次路径；执行模式直接将成功结果交给 `/follow_path`。只有真实 `FollowPath` 成功才发布保留的 `/navigation_complete`（`std_msgs/msg/Empty`）；失败、取消和只规划模式不发布到达事件。
- 只规划模式不启动控制器，也不创建跟踪动作客户端。失败、取消与真正到达分别记录，不自动重试、重规划或执行恢复动作。
- 当前采用 `RotationShimController` 对齐起步朝向，再由 DWB 的 `LimitedAccelGenerator` 执行 DWA 控制。速度、预测和评分参数集中在 `nav2.yaml`，目录迁移不等于跟踪算法已经验收。
- 机器人折叠姿态碰撞投影半径约 0.442 m，建议使用含余量的 0.50 m 圆形轮廓；不能为通过窄道缩小到真实几何以下，伸臂后也不能继续使用此固定轮廓。当前参数以 `nav2.yaml` 为准，用户自行调整时不擅自覆盖。
- 全局和局部代价地图都使用 `map` 坐标，局部为滚动窗口，静态层保留桌面投影，激光层提供当前观测。速度同时受轴向限制与平面速度限制约束，不能只增大 `max_vel_x` 而忽略 `max_speed_xy`。编辑 YAML 后需重新启动相关入口。
- `/goal_pose`、`/global_plan` 保留最后一条消息，RViz 晚启动也可显示；`/local_plan` 显示当前局部控制选择，不是已走轨迹。独立只规划模式共用导航视图，未启动的局部图层可能没有数据。
- 退出时在 ROS 上下文仍有效时，为目标接收、取消和结果共用最多两秒等待；速度由 Nav2 控制器发布。主入口遇到仿真、控制器或任务节点退出会统一结束，不留下无管理运动进程。

## 验证记录

- 此前配置/launch 整理时，两个受影响包构建通过，150 个原配置参数与运行节点的有效值一致。独立规划得到有效目标路径且不驱动机器人；一次并行仿真遇到 Fast DDS 服务响应超时，隔离后通过，未在代码加入重试。
- 此前导航精度检查曾在 Nav2 返回成功后测得停稳位姿略超容差；该记录不等于当前参数的复测结果。本次抓放不调整导航参数，也不声称解决了所有导航精度问题。
- 抓放阶段：三个受影响包构建通过，三个独立 IK 边界测试通过。已从出生点实跑完整主入口，没有预设机器人到工作台、没有移动物块坐标或添加附着约束；导航前机械臂保持收拢，导航成功后按六阶段执行。物块实测提升约 0.148 m，最后位于 `(3.19174, -2.02549, 0.76000)` 的托盘内，七轴最大复位误差约 0.00050 rad。Gazebo 与 RViz 实际界面均已观察。


## 真值 IK 抓放

- `manipulation_pkg/manipulation_pkg/pick_place_node.py` 是唯一抓放节点，顺序执行：打开并接近物块、夹紧、提升、移至托盘并下降、松爪退离、恢复抓取前关节姿态。这里“原点”是机械臂操作前的七轴关节姿态，不是底盘出生点。
- `kinematics.py` 从 `/robot_description` 的展开 URDF 读取安装座、七个转动关节和 TCP 固定变换，不另维护 DH 表。系统 NumPy/SciPy 做有界数值 IK，独立检查位置残差 1 mm、角度残差 0.01 rad；失败明确报告，不返回近似成功。
- `/ground_truth/poses` 来自已有 SceneBroadcaster 的 `/world/workcell/pose/info`，桥接配置在 `auto_pkg/config/bridge.yaml`。模型位姿相对 `world`，链接位姿相对模型；单个消息缺少父帧/有效时间戳，不能直接灌入 `/tf`。操作节点显式组合这些位姿，不用 AMCL 或出生坐标冒充物块/夹爪真值。
- Gazebo 合并了手掌及 TCP 的固定关节；实际 TCP 由物理 `fer_link7` 真值乘 URDF 固定末端变换得到。七轴反馈来自 `/joint_states`，命令使用 `/arm_controller/follow_joint_trajectory`；双指通过 `/gripper_controller/commands` 发送同号力命令，单位 N。
- 放置目标读取场景托盘底板几何并组合工作台真值，不复制一份目标坐标。夹持依靠原有接触与摩擦，不调用物块 `set_pose` 或附着服务。允许物块受重后落座，检查提升高度和夹持稳定性，按真实夹持变换及物块旋转投影计算下放高度，松爪后确认物块仍在托盘内。
- 当前使用适合本场景的倾斜抓取、上方接近、分段 Cartesian IK 与关节轨迹；不是通用机械臂碰撞规划器，不保证任意场景和目标都可达。修改工作区几何或抓取姿态后须实际验证。
- 独立入口默认 `wait_for_navigation:=false`，需先把机器人停在工作台旁；完整主入口传 `true`，只接受导航成功事件。操作期间发布零 `/cmd_vel` 保持底盘停车，等待导航期间绝不发布，完成后释放停车定时器；不要在操作中手动驾驶或重发导航目标。
- 进度发布为保留的 `/manipulation_state`：`waiting/grasp/close/lift/place/open/return/complete/failed`。查看当前状态可用 `ros2 topic echo /manipulation_state --qos-durability transient_local --once`。
- `Ctrl+C` 在 ROS 上下文有效时取消机械臂动作，最多等待两秒；若正在夹持，不自动松爪。失败保留明确状态和停车保持，不自动重试或继续后续阶段，须先安全处理夹持物块。

## 维护约定

- `README.md` 由用户维护，不修改本项目或上级 README；项目说明和代理约定写在本文件。
- 只有用户明确要求时才提交或推送 Git，默认保留未提交改动。
- 不创建空节点、接口壳、通用基类或配置管理框架。必要的激光有效值/越界处理、时间戳 TF、仿真时钟、控制器启动依赖和取消停车不能作为“防御代码”删掉。
- 结构重构与算法调参分开；以实际运行的路径、目标位置/朝向和停车结果验收，不把启动成功或规划成功当作完成导航。
- `build/`、`install/`、`log/`、`__pycache__/` 是生成物，不是源码。只清理已废弃入口对应的生成物，不删除用户其他工作。
