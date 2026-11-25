from glob import glob
from setuptools import find_packages, setup

package_name = "auto_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/asset", glob("asset/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="long753",
    maintainer_email="cl2848077709@outlook.com",
    description="Workcell simulation and application bringup",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "moving_obstacle_node = auto_pkg.moving_obstacle_node:main",
        ],
    },
)
