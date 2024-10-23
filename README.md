# STERLING

## Prerequisites
Before you begin, ensure you have the following:

1. **ROS 2 Installed**: Make sure you have ROS 2 Humble installed on your system. Follow the official [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html) for your operating system.
2. **Create a ROS 2 Workspace**:
    - Open a terminal and create a directory for your ROS 2 workspace:
      ```sh
      mkdir -p ~/utexas_ws/src
      ```
3. **Clone the Repository**:
    - Clone this repository into the `src` folder of your workspace:
      ```sh
      cd ~/utexas_ws/src
      git clone git@github.com:UTNuclearRobotics/utexas_sterling.git
      ```
4. **Install Dependencies**:
    - Use `rosdep` to install dependencies:
      ```sh
      cd ~/utexas_ws
      rosdep update
      rosdep install --from-paths src --ignore-src -r -y
      ```
5. **Build the Workspace**:
    - Us `colcon` to build the workspace:
      ```sh
      colcon build
      ```
6. **Source the Workspace**:
    - Source the setup file to overlay this workspace on your environment:
      ```sh
      source install/setup.bash
      ```

# Workflow
The workflow consists of three main phases:

1. **Data Collection**: Record sensor data from a robot into a rosbag.
2. **Offline Preprocessing**: The rosbag data is converted into a Python dictionary format used for PyTorch model training. TODO: add more description.
3. **Deployment**: Deploy the trained models to the robot for terrain-aware autonomous navigation.

## Recording Rosbag
Record sensor data from a robot into a rosbag. Update the topic names in the configuration file. To start recording with the specified parameters, use the following command:

```sh
ros2 launch visual_representation_learning record_rosbag.launch.py
```

### Parameters
- `bag_name`: Name of the ROS bag to save in the `bags` directory.

### Output
- Recorded rosbag is saved in `bags` directory in the top level workspace directory.

### Files
- visual_representation_learning/config/`rosbag.yaml`
- visual_representation_learning/launch/`record_rosbag.launch.py`

## Process Rosbag
Convert the recorded rosbag data into a Python dictionary format suitable for PyTorch training. To launch the `process_rosbag` with the necessary parameters, use the following command:

```sh
ros2 launch visual_representation_learning process_rosbag.launch.py
```

### Parameters
- `bag_name`: Name of the ROS bag to process in the `bags` directory.
- `visual`: Set to `true` to enable graphical feedback of data.

### Output
- Processed rosbag is saved in `datasets` directory in the top level workspace directory.

### Files
- visual_representation_learning/config/`rosbag.yaml`
- visual_representation_learning/launch/`process_rosbag.launch.py`
- visual_representation_learning/visual_representation_learning/`process_rosbag.py`

## Train Terrain Representations
Convert pickle files into tensors and train the terrain representations using a PyTorch script. To start the training process, use the command:

```sh
ros2 run visual_representation_learning train_auto_encoder
```

### Parameters
- `--config`: Name of the `yaml` file that defines training and validation pickle file datasets.

### Output
- Checkpoint (`.ckpt`) file is saved in `torch/terrain_representations/checkpoints`.
- Model (`.pt`) file is saved in `torch/terrain_representations/models`.
- a `istat.yaml`

### Files
- visual_representation_learning/config/`dataset.yaml`
- visual_representation_learning/visual_representation_learning/train/terrain_representations/`data_loader.py`
- visual_representation_learning/visual_representation_learning/train/terrain_representations/`train_auto_encoder.py`