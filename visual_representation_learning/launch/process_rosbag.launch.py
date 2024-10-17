#!/usr/bin/env python3
"""
process_rosbag.launch.py

This script sets up and launches a ROS2 process to preprocess the rosbag file into
a pickle file to feed into the PyTorch model.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Set the environment variable to change the logging format
os.environ["RCUTILS_CONSOLE_OUTPUT_FORMAT"] = "[{severity}] [{name}]: {message}"

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))


def launch_setup(context, *args, **kwargs):
    bag_path = os.path.join(ros_ws_dir, "bags", LaunchConfiguration("bag_name").perform(context))
    save_path = os.path.join(ros_ws_dir, "pickles")

    return [
        Node(
            package="visual_representation_learning",
            executable="process_rosbag",
            name="process_rosbag",
            output="screen",
            parameters=[
                {
                    "bag_path": bag_path,
                    "save_path": save_path,
                    "visual": LaunchConfiguration("visual"),
                }
            ],
        )
    ]


def generate_launch_description():
    declared_arguments = []

    # Declare the launch arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "bag_name",
            default_value="",
            description="Name of the ROS bag folder containing the bag files",
        ),
    )

    declared_arguments.append(
        DeclareLaunchArgument("visual", default_value="false", description="Whether to visualize the results"),
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
