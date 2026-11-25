# 机器人自主避障运动与作业规划仿真

## 实验准备
### 配置环境
使用 Ubuntu 22.04 的系统 Python（`/usr/bin/python3`），依赖通过 APT 安装的 ROS 2 Humble 和 Gazebo Fortress。项目的构建代码如下：
```bash
ros_humble
colcon build
source install/setuo.sh
```

### 构建地图
启动主仿真(Terminal 1)：
```bash
ros2 launch auto_pkg world.launch.py
```

开始建图(Terminal 2)：
```bash
ros2 launch navigation_pkg mapping.launch.py
```

移动机器人(Terminal 3)：
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
当地图构建完成后在 Terminal 2 中输入 `ctrl+c` 将地图保存到 outputs/mapping 里面


## 实验
### 核心实验一：导航算法对比
**有地图，无动态障碍物：**
```bash
ros2 launch auto_pkg main_with_map.launch.py 
```

**有地图，有动态障碍物：**
```bash
ros2 launch auto_pkg main_with_map.launch.py 
```

**无地图，无动态障碍物：**
```bash
ros2 launch auto_pkg main_without_map.launch.py 
```

**无地图，有动态障碍物：**
```bash
ros2 launch auto_pkg main_without_map.launch.py 
```

### 核心实验二：控制算法对比

### 辅助实验一：路径规划算法
该实验用于对比各类路径规划算法所生成的路径，最终会将所有算法生成的路径图像存放在 `outputs/compare_navigation` 中

对比算法包含：A*、Dijkstra、Theta*、RRT*

启动文件为：
```bash
python3 script/compare_navigation.py
```

### 辅助实验二：速度控制算法
启动文件为：
```bash
python3 script/compare_cmd.py
```

## 任务原理
本任务基于 `ros2 humble`, `python 3.10.12`, `gazebo`, `rviz` 构建。核心代码存储在 `src/auto_pkg` 中（使用 ros2 的标准架构）

**任务流程：**
1. 使用 gazebo 搭建任务环境
2. 直接调用 gazebo 中的摄像机位姿真值，并使用 2D slam 构建环境地图（这里图方便直接调用的真值，实际可能需要使用 IMU 积分+重定位来获得摄像机位姿）
3. 使用路径规划算法（实时 or 地图先验）控制机器人的移动
4. 设计机械臂操作流程（）
5. 使用标准 IK 计算机械臂的关节角变化
