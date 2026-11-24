from setuptools import find_packages, setup

package_name = "auto_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="long753",
    maintainer_email="cl2848077709@outlook.com",
    description="ROS 2 mobile manipulation exercises",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "navigation = auto_pkg.navigation_node:main",
        ],
    },
)
