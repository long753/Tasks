# 机器人自主避障运动与作业规划仿真

## Setup
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

### 正式测试
**无动态障碍物：**
```bash
ros2 launch auto_pkg main_with_map.launch.py 
```

**有动态障碍物：**
```bash
ros2 launch auto_pkg main_with_map.launch.py moving_obstacle:=true
```


## 任务原理
本任务基于 `ros2 humble`, `python 3.10.12`, `gazebo`, `rviz` 构建。核心代码存储在 `src/auto_pkg` 中（使用 ros2 的标准架构）

**任务流程：**
1. 使用 gazebo 搭建任务环境
2. 直接调用 gazebo 中的摄像机位姿真值，并使用 2D SLAM 构建环境地图（这里图方便直接调用的真值，实际可能需要使用 IMU 积分+重定位来获得摄像机位姿）
3. 使用路径规划算法控制机器人的移动
4. 使用标准 IK 计算机械臂的关节角变化
5. 设计机械臂操作流程（移动到物块位置 -> 夹爪加紧 -> 提升物块 -> 移动到目标位置 -> 夹爪松开 -> 机械臂复位）

**算法原理：**
导航算法：使用 A* 构建全局导航路径，使用 DWB 实现实时避障与速度控制
机械臂控制算法： 

**图片描述：**
`outputs/images/1.png`：由于轮胎打滑导致的 SLAM 定位偏移
`outputs/images/2.png`：代价地图与全局导航路径
`outputs/images/3.png`：机械臂操作第一步，移动到物块位置
`outputs/images/4.png`：机械臂操作第二步，夹爪加紧
`outputs/images/5.png`：机械臂操作第三步，提升物块
`outputs/images/6.png`：机械臂操作第四步，移动到目标位置
`outputs/images/7.png`：机械臂操作第五步，夹爪松开
`outputs/images/8.png`：机械臂操作第六步，机械臂复位
`outputs/main.webm`：整个任务的视频录像
