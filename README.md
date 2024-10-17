# STERLING

## Prerequisites
Ensure you have ROS 2 installed and properly set up. Create a `utexas_sterling_ws` ROS workspace in your `$HOME` directory and clone this repository in its `src` folder.

## Running Scripts
To launch the `process_rosbag` with the specified parameters, use the following command:

```sh
ros2 launch visual_representation_learning process_rosbag.launch.py bag_path:=philbart_sample/ visual:=true
```

## Parameters
- `bag_path`: Path to the ROS bag file or directory.
- `visual`: Set to `true` to enable visual processing.