#!/usr/bin/env bash
set -eo pipefail

if [[ ${1:-} != ros2 && ${1:-} != python ]]; then
    printf 'Usage: bash %s {ros2|python} [args...]\n' "$0" >&2
    exit 2
fi

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ros=/opt/ros/humble
if [[ ! -r "$ros/setup.bash" ]]; then
    printf 'ROS 2 Humble APT installation not found: %s/setup.bash\n' "$ros" >&2
    exit 1
fi

# Do not import Python packages or ROS overlays from the caller's environment.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV AMENT_PREFIX_PATH COLCON_PREFIX_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT="$project/.venv"
source "$ros/setup.bash"

if [[ $1 == ros2 ]]; then
    shift
    set -- python "$ros/bin/ros2" "$@"
fi
exec uv run --project "$project" --locked "$@"
