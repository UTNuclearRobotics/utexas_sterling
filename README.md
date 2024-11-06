# Project Setup

## Prerequisites

- **ROS 2 Humble**: Ensure you have ROS 2 Humble installed on your system. Follow the official [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html) for your operating system.
- **Python 3.8+**: Ensure you have Python 3.8 or higher installed.

## ROS 2

Install the following ROS2 packages:

```sh
sudo apt install ros-humble-rosbag2-py
```

## Python Virtual Environment

Run the scripts from inside a virtual environment.

1. Create the virtual environment and install dependencies:

```sh
chmod +x setup_venv.sh
./setup_venv.sh
```

2. To enter the virtual environment with ROS sourced:

```sh
source /opt/ros/humble/setup.bash
source venv/bin/activate
```