from glob import glob

from setuptools import find_packages, setup

package_name = "manipulation_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/asset", ["asset/workcell.sdf"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="long753",
    maintainer_email="cl2848077709@outlook.com",
    description="Direct-IK pick and place for the workcell.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pick_place_node = manipulation_pkg.pick_place_node:main",
        ],
    },
)
