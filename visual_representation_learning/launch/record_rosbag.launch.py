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

# Load the configuration from the YAML file
with open(os.path.join(package_share_directory, "config", "rosbag.yaml"), "r") as file:
    config = yaml.safe_load(file)
    TOPIC_DICT = config["recorded_topics"]

def launch_setup(context, *args, **kwargs):
    topics_list = list(TOPIC_DICT.values())
    
    bag_name = LaunchConfiguration("bag_name").perform(context)
    output_dir = LaunchConfiguration("output_dir").perform(context)
    bag_file_path = os.path.join(output_dir, bag_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the bag file already exists
    if os.path.exists(bag_file_path):
        raise FileExistsError(f"The bag file '{bag_file_path}' already exists")

    # Define the process to record the bag
    record_bag_process = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", bag_file_path] + topics_list,
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
                    LogInfo(msg=f"Bags are stored at the directory: \033[0;33m{output_dir}\033[0m"),
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

    declared_arguments.append(
        DeclareLaunchArgument(
            "output_dir",
            default_value=os.path.expanduser("~/utexas_sterling_ws/bags"),
            description="Directory to store generated bag files",
        ),
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
