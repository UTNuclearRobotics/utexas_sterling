# Offline Preprocessing Workflow

## Process Rosbag

Convert the recorded rosbag data into a Python dictionary format suitable for PyTorch training.

### Command

```sh
python3 process_rosbag.py
```

### Parameters
- `bag_path`: Path to the ROS2 bag file.
- `config_path`: Path to the config file. Default is `config/rosbag.yaml`.
- `save_path`: Path to save the processed data. Default is `../../datasets`.
- `visual`: Enable visualization. Default is `false`.

### Output

Processed rosbag is saved in the `datasets` directory.

## Train Representations

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