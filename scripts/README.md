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

## 3. Calibrate Homography
...

## 4. Create VICReg Dataset
Use the synchronized data to get a list of `N` viewpoints of a terrain image patch.

### Command
```sh
python3 vicreg_dataset.py -b <bag_dir>
```

### Parameters
- `-b`: Path to the ROS bag folder. Should contain a `<bag_dir>_sync.pkl` file.

### Output
The VICReg dataset pickle is saved in the specified bag directory as `<bag_dir>_vicreg.pkl`.

## 5. Train Terrain Representations
Train a PyTorch model on classifying terrain.

### Command
```sh
python3 train_representations.py -b <bag_dir>
```

### Parameters
- `-b`: Path to the ROS bag folder. Should contain a `<bag_dir>_vicreg.pkl` file.
- `-batch_size`: Input batch size for training.
- `-epochs`: Number of epochs to train.

### Output
The PyTorch model will be in a `/models` folder in the `<bag_dir>` as `terrain_rep.pt`.

# 6. Create Terrain Clusters
Configure the number of clusters and saves a visual sample of each.

### Command
```sh
python3 cluster_ui.py
```

### Inputs
The UI will ask you to locate 2 files:
- `<bag_dir>/vicreg.pkl`
- `<bag_dir>/models/terrain_rep.pt`

### Output
Saves the centroid of each cluster as `kmeans_model.pkl`, `scalar.pkl`, and `preference.yaml` under the `<bag_dir>/clusters` directory.

# 7. Create BEV Costmap Video
Creates a video of a BEV costmap stacked on top of a BEV of the raw camera feed.

### Parameters
- `-b`: Path to the ROS bag folder. Should contain a `<bag_dir>_vicreg.pkl` file.
...

### Output
Uses saves the video as `costmap.mp4`.
