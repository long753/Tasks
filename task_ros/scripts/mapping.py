"""Build a 2D map with SLAM Toolbox and save it on Ctrl+C."""

from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory


OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "map"


def save_map(saver: Path) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Do not replace the previous map if Nav2 times out or fails to write a pair.
    with TemporaryDirectory(dir=OUTPUT) as temp_dir:
        prefix = Path(temp_dir) / "map"
        result = subprocess.run(
            [
                saver, "-t", "/map", "-f", prefix, "--fmt", "pgm",
                "--ros-args", "-p", "map_subscribe_transient_local:=true",
                "-p", "save_map_timeout:=8.0",
            ],
            start_new_session=True,
            timeout=15,
        )
        files = [prefix.with_suffix(".pgm"), prefix.with_suffix(".yaml")]
        if result.returncode or not all(f.is_file() and f.stat().st_size for f in files):
            raise RuntimeError("没有保存到有效地图。请保持仿真运行，检查 /scan 和 odom → base_link → laser_link。")
        for file in files:
            file.replace(OUTPUT / file.name)


def main() -> int:
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_prefix
    except ImportError:
        print("请先执行 source /opt/ros/humble/setup.bash，再用系统 python3 运行。", file=sys.stderr)
        return 1

    try:
        toolbox = Path(get_package_prefix("slam_toolbox"))
        nav2 = Path(get_package_prefix("nav2_map_server"))
    except PackageNotFoundError as error:
        print(f"{error}\n请执行 sudo apt install ros-humble-slam-toolbox，然后重新加载 ROS 环境。", file=sys.stderr)
        return 1

    params = toolbox / "share/slam_toolbox/config/mapper_params_online_async.yaml"
    command = [
        toolbox / "lib/slam_toolbox/async_slam_toolbox_node",
        "--ros-args", "--params-file", params,
        "-p", "use_sim_time:=true",
        "-p", "base_frame:=base_link",
        "-p", "max_laser_range:=12.0",
        "-p", "map_update_interval:=1.0",
        "-p", "minimum_travel_distance:=0.20",
        "-p", "minimum_travel_heading:=0.20",
        "-p", "use_map_saver:=false",
    ]
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        # Keep SLAM alive after the terminal sends Ctrl+C to this script.
        slam = subprocess.Popen(command, start_new_session=True)
        try:
            print("SLAM 已启动：请在另一终端控制底盘。完成后在这里按 Ctrl+C 保存。", flush=True)
            while not stopping:
                try:
                    status = slam.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    continue
                raise RuntimeError(f"SLAM Toolbox 提前退出（状态：{status}）。")

            print("正在保存 /map，请不要关闭仿真……", flush=True)
            save_map(nav2 / "lib/nav2_map_server/map_saver_cli")
            print(f"地图已保存：{OUTPUT / 'map.yaml'} 和 map.pgm", flush=True)
        finally:
            slam.terminate()
            try:
                slam.wait(timeout=5)
            except subprocess.TimeoutExpired:
                slam.kill()
                slam.wait()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"建图失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())