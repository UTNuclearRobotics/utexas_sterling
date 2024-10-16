#!/usr/bin/env python3
"""
record_rosbag.launch.py

This script sets up and launches a ROS2 process to record specified topics into a rosbag file.
The configuration for the topics to be recorded is loaded from a YAML file.
"""

import os
from datetime import datetime

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))

# Load the configuration from the YAML file
with open(os.path.join(package_share_directory, "config", "rosbag.yaml"), "r") as file:
    config = yaml.safe_load(file)
    TOPIC_DICT = config["recorded_topics"]


def launch_setup(context, *args, **kwargs):
    topics_list = list(TOPIC_DICT.values())

    save_path = os.path.join(ros_ws_dir, "bags")
    bag_name = LaunchConfiguration("bag_name").perform(context)
    bag_save_file = os.path.join(save_path, bag_name)

    # Ensure the output directory exists
    os.makedirs(save_path, exist_ok=True)

    # Check if the bag file already exists
    if os.path.exists(bag_save_file):
        raise FileExistsError(f"The bag file '{bag_save_file}' already exists")

    # Define the process to record the bag
    record_bag_process = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", bag_save_file] + topics_list,
        output="screen",
    )

    return [
        LogInfo(
            msg=f"Recording topics: \033[0;33m{topics_list}\033[0m",
        ),
        record_bag_process,
        RegisterEventHandler(
            OnProcessExit(
                target_action=record_bag_process,
                on_exit=[
                    LogInfo(msg=f"Bags are stored at: \033[0;33m{save_path}\033[0m"),
                    LogInfo(msg=f"Bag file name: \033[0;33m{bag_name}\033[0m"),
                ],
            )
        ),
    ]


def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "bag_name",
            default_value=f'vrl_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            description="Name of the bag file to save the recording",
        ),
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
