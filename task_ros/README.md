# 机器人自主避障运动与作业规划仿真

## setup
### 配置环境

使用 Ubuntu 22.04 的系统 Python（`/usr/bin/python3`），依赖通过 APT 安装的 ROS 2 Humble 和 Gazebo Fortress。

### 构建地图
启动主仿真(Terminal 1)：
```bash
ros2 launch scripts/main.launch.py
```

启动 slam(Terminal 2)：
```bash
ros2 launch scripts/mapping.launch.py
```

移动机器人(Terminal 3)：
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
当地图构建完成后在 Terminal 2 中输入 `ctrl+c` 将地图保存到 outputs/mapping 里面


## 有地图
### 启动主程序
```bash
ros2 launch scripts/main_with_map.launch.py
```


## 无地图
### 启动主程序
```bash
ros2 launch scripts/main_without_map.launch.py
```


## 任务原理
本任务基于 `ros2 humble`, `python 3.10.12`, `gazebo`, `rviz` 构建。核心代码存储在 `src/auto_pkg` 中（使用 ros2 的标准架构）

**任务流程：**
1. 使用 gazebo 搭建任务环境
2. 直接调用 gazebo 中的摄像机位姿真值，并使用 2D slam 构建环境地图（这里图方便直接调用的真值，实际可能需要使用 IMU 积分+重定位来获得摄像机位姿）
3. 使用路径规划算法（实时 or 地图先验）控制机器人的移动
4. 设计机械臂操作流程（）
5. 使用标准 IK 计算机械臂的关节角变化

**路径规划算法对比实验：**
- 有地图，无动态障碍物：
- 有地图，有动态障碍物：
- 无地图，无动态障碍物：
- 无地图，有动态障碍物：

定位算法：`AMCL`, `SLAM`
全局规划算法：`A*`
局部避障算法：`DWA`, 

## 图像说明
`/outputs/images/1.png`: 由于轮胎打滑导致的 AMCL 定位偏差