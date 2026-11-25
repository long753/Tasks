"""Read the installed workcell scene; its XY coordinates match the saved map."""

from pathlib import Path
from xml.etree import ElementTree

from ament_index_python.packages import get_package_share_directory


def scene_file():
    return Path(get_package_share_directory("navigation_pkg")) / "asset" / "workcell.sdf"


def marker_pose(name):
    world = ElementTree.parse(scene_file())
    return tuple(map(float, world.findtext(f"./world/model[@name='{name}']/pose").split()))
