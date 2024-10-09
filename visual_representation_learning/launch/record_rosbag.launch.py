#!/usr/bin/env python3
"""
record_rosbag.launch.py

This launch file is used to record ROS2 topics into a rosbag file. The topics to be recorded
are specified in a YAML configuration file. The recorded bag files are stored in a directory
with a timestamped filename to avoid overwriting. The launch file includes functionality to
declare launch arguments, execute the recording process, and log information about the recording.

Functions:
- load_topics: Loads the list of topics to be recorded from a YAML configuration file.
- get_file_name: Generates a timestamped filename for the rosbag and ensures the directory exists.
- launch_setup: Sets up the recording process and logs relevant information.
- generate_launch_description: Declares launch arguments and returns the launch description.

Usage:
- The topics to be recorded should be specified in 'record_rosbag.yaml' located in the config directory.
- The recorded bag files will be stored in the 'bags' directory with a timestamped filename.
"""

import os
from datetime import datetime

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def load_topics():
    # Create the full path to the config file directory
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")

    # Load the default topics from the YAML file
    config_file = os.path.join(config_dir, "record_rosbag.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config["topics"]


def get_file_name():
    # Create the full path to the bags file directory
    bag_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "bags"))

    # Ensure the directory exists
    os.makedirs(bag_dir, exist_ok=True)

    # Generate the timestamp and bag file name to ensure no overwriting of bags
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return bag_dir, f"vrl_data_{timestamp}"


def launch_setup(context, *args, **kwargs):
    topics_list = load_topics()
    bag_file_path, _ = get_file_name()
    bag_name = LaunchConfiguration("bag_name").perform(context)

    # Define the process to record the bag
    record_bag_process = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            PathJoinSubstitution([bag_file_path, bag_name]),
        ]
        + topics_list,
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
                    LogInfo(
                        msg=f"Bags are stored at the directory: \033[0;33m{bag_file_path[bag_file_path.find('install'):]}\033[0m"
                    ),
                    LogInfo(msg=f"Bag file name: \033[0;33m{bag_name}\033[0m"),
                ],
            )
        ),
    ]


def generate_launch_description():
    declared_arguments = []
    _, default_bag_name = get_file_name()

    declared_arguments.append(
        DeclareLaunchArgument(
            "bag_name",
            default_value=default_bag_name,
            description="Name of the bag file to save the recording",
        ),
    )

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
