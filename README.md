# STERLING

## Prerequisites
Ensure you have ROS 2 installed and properly set up. Create a ROS workspace and clone this repository in its `src` folder. Install dependencies with `rosdep` and build the workspace with `colcon build`. Source this workspace with `install/setup.bash` before running any commands.

# Workflow

## Recording Rosbag
Record sensor data from a robot into a rosbag. Update the topic names in the configuration file. To start recording with the specified parameters, use the following command:

```sh
ros2 launch visual_representation_learning record_rosbag.launch.py
```

### Parameters
- `bag_name`: Name of the ROS bag to save in `bags` directory.

### Files
- config/rosbag.yaml
- launch/record_rosbag.launch.py

## Process Rosbag
Convert the recorded rosbag data into a Python dictionary format suitable for PyTorch training. To launch the `process_rosbag` with the necessary parameters, use the following command:

```sh
ros2 launch visual_representation_learning process_rosbag.launch.py
```

### Parameters
- `bag_name`: Name of the ROS bag to process in the `bags` directory.
- `visual`: Set to `true` to enable graphical feedback of data.

### Files
- config/rosbag.yaml
- launch/process_rosbag.launch.py
- visual_representation_learning/process_rosbag.py