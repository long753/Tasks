from glob import glob

from setuptools import find_packages, setup

package_name = "navigation_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/asset", ["asset/workcell.sdf"]),
        (
            "share/" + package_name + "/maps",
            ["maps/map.yaml", "maps/map.png"],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/nav2.yaml",
                "config/mapping.rviz",
                "config/navigation.rviz",
            ],
        ),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="long753",
    maintainer_email="cl2848077709@outlook.com",
    description="Ground-truth mapping and fixed-map Nav2 navigation for the workcell.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mapping_node = navigation_pkg.mapping_node:main",
            "planning_map_node = navigation_pkg.planning_map_node:main",
            "navigation_task_node = navigation_pkg.navigation_task_node:main",
        ],
    },
)
