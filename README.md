# STERLING

## Prerequisites
Ensure you have ROS 2 installed and properly set up. Create a ROS workspace and clone this repository in its `src` folder. Install dependencies with `rosdep` and build the worksapce with `colcon build`.

## Recording Rosbag
To launch the `record_rosbag` with parameters, use the following command:

```sh
ros2 launch visual_representation_learning record_rosbag.launch.py
```

### Parameters
- `bag_name`: Name of the ROS bag to save in `bags` directory.

## Process Rosbag

To launch the `process_rosbag` with parameters, use the following command:

```sh
ros2 launch visual_representation_learning process_rosbag.launch.py bag_name:=philbart_sample/ visual:=true
```

## Parameters
- `bag_name`: Name of the ROS bag to process in the `bags` directory.
- `visual`: Set to `true` to enable graphical feedback of data.