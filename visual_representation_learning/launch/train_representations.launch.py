#!/usr/bin/env python3
"""
train_representations.launch.py

This script sets up and launches a ROS2 process to train representations using
a PyTorch model.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess


# Set the environment variable to change the logging format
os.environ["RCUTILS_CONSOLE_OUTPUT_FORMAT"] = "[{severity}] [{name}]: {message}"

package_share_directory = get_package_share_directory("visual_representation_learning")

def launch_setup(context, *args, **kwargs):
    config = LaunchConfiguration("config").perform(context)

    return [
        ExecuteProcess(
            cmd=[
                "train_auto_encoder",
                "--config",
                config,
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    declared_arguments = []

    # Declare the launch arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "config",
            default_value=os.path.join(package_share_directory, "config", "dataset.yaml"),
            description="Data config file defining which datasets to use for training and validation",
        ),
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
