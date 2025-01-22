# UTexas Sterling Workflow

## 1. Collect ROS Bag

Record data to a ROS bag. Include the following topics:
- `tf2_msgs/TFMessage` (Know static transform between odometry and camera frames)
- `sensor_msgs/msg/CompressedImage` (Camera RGB)
- `sensor_msgs/msg/CameraInfo` (don't have to record, but needed for camera intrinsics)
- `nav_msgs/msg/Odometry`
- `sensor_msgs/msg/Imu`

Add the recorded ROS bag to the `bags` directory.

## 2. Synchronize the ROS Bag

Convert the recorded ROS bag into a Python dictionary format. Synchronize messages by timestamp.

### Command

```sh
python3 synchronize_rosbag.py -b <bag_dir>
```

### Parameters
- `-b`: Path to the ROS bag folder. Should contain a `metadata.yaml` and `.db3` file.
- `-v`: Save video of processed ROS bag. Default is `false`.

### Output

The synchronized ROS bag is saved in the specified bag directory as `<bag_dir>_synced.pkl`. Each message is synchronized to the closest timestamp, resulting in lists of `image`, `imu`, and `odom` messages. For example, index `N` of each list contains messages that correspond to similar timestamps.

## 3. Format VICReg Dataset



Define what pickle files to use as the datasets for training and validating in a `data_config.yaml`

### Command

```sh
python3 train_representations.py
```

### Parameters
- `batch_size`: Input batch size for training. Default is `512`.
- `epochs`: Number of epochs to train. Default is `200`.
- `lr`: Learning rate. Default is `3e-4`.
- `l1_coeff`: L1 loss coefficient. Default is `0.5`.
- `num_gpus`: Number of GPUs to use. Default is `1`.
- `latent_size`: Size of the common latent space. Default is `128`.
- `imu_in_rep`: Whether to include the inertial data in the representation. Default is `1`.
- `data_config_path`: Path to data config file. Default is `config/data_config.yaml`.

### Output
- Saved files will be in a `/models` folder in the format `rep_(accuracy)_(date)_(time)`. Files include:
  - Encoder weights (`.pt`)
  - Cluster images (`.png`)