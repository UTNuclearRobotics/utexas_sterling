# Camera Calibration Guide

## Calibrating homography matrix from chessboard image

### Setting up chessboard

1. Ensure the chessboard being used to calibrate the homography matrix has known cell size. Note the number of rows and columns as well. The calibration chessboard we are using is 11 rows x 10 columns with 10cm cells.
2. Teleop the vehicle to a flat surface. This step is critical for capturing a "true" ground plane from the chessboard.
3. View live image stream from camera using RVIZ or Mapviz and move chessboard so that it appears to align with the bottom of the image and is centered in the image, then move it away from this point / away from the vehicle by 0.5m. The location of the chessboard doesn't have to be exact. The area captured in the BEV image can be adjusted by changing the grid size parameter in sterling_patern_deployment, and validating it in full_bev.py.
4. Record/save image of chessboard as .png file in the following directory: /sterling_patern_offline/scripts/homography/
5. Create config.yaml file with camera intrinsics values in the following directory: /sterling_patern_offline/scripts/homography/<br>
    The format should look like this:<br>
    camera_intrinsics: {cx: 637.9826049804688, cy: 358.5901794433594, fx: 759.8603515625,
  fy: 759.8603515625}
6. You may get an error about the cv2.Window function not being implemented. Run the following:<br>
'''<br>
sudo apt-get update<br>
sudo apt-get install -y libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev<br>
pip uninstall opencv-python opencv-contrib-python -y<br>
pip install opencv-contrib-python opencv-python<br>
'''

### Calibrating homography matrix: homography_from_chessboard.py
The following script will calibrate the homography matrix from the image and save the homography matrix, plane_distance, plane_normal, and rigid_transform in config.yaml<br>
'''<br>
python3 scripts/homography_from_chessboard.py --image_path /scripts/homography/calibration.png --rows 10 --cols 10 --validate

Args:<br>
--image_path, -i = path to chessboard image<br>
--rows, -r =  Number of rows in the chessboard<br>
--cols, -c = Number of columns in the chessboard<br>
--validate, -v = Visualize detected chessboard points on the image<br>
'''

WARNING:<br>
If the chessboard used for calibration is square, the homography matrix may need to be rotated because the detected corner coordinates are in the wrong order. The homography matrix can be validated using the python3 full_bev.py. Change the image_file variable in main to your calibration image name.<br>

'''<br>
self.H, mask = cv2.findHomography(model_chessboard_2d, self.corners, cv2.RANSAC)<br>
self.R_ccw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)<br>
self.R_cw = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)<br>
self.H = self.H @ self.R_ccw<br>
'''

# Creating Datasets and Training Guide

## Synchronizing data from rosbag: synchronize_rosbag.py

1. Change topic names for odom and imu to the recorded rosbag topics in __init__ self.odometry_topic and self.imu_topic.

'''<br>
python3 scripts/synchronize_rosbag.py --bag_path <bag_dir> --save_path <bag_dir> --visual --simulation --threshold <threshold> --skip_last_n <skip_n> --save_interval <save_n>

Args:
--bag_path, -b = Path to rosbag folder<br>
--save_path, -s =  Optional, path to save synced file. If not used, it saves it to rosbag folder which is typical. Default is None.<br>
--visual, -v = Action. If used, creates an MP4 in rosbag folder<br>
--simulation, -sim = Action. Used for differentiating sim and real topics.<br>
--threshold, -th = Specifies the time threshold used to determine the window of time the image, IMU,and odometry messages from the bag file must be to be considered as part of the same timestep and be processed together.<br>
--skip_last_n, -n = Removes n number of timesteps from synced file. Used to filter out data when vehicle is stationary for improved training. Default is 0.<br>
--save_interval, int = Saves synchronized data every n timesteps. Used to reduce synced file size. Default is 0.<br>
'''

## Creating vicreg dataset: vicreg_dataset.py

1. Change camera_odom_offset variable with actual transform from base_link to camera.

'''<br>
python3 vicreg_dataset.py -b <bag_dir>

Args:<br>
-b = Path to the directory containing "<bag_dir>_synced.h5"<br> 
'''

## STERLING Training: train_representation.py

'''<br>
python3 train_representation.py -b <bag_dir> -batch_size <batch_size> -epochs <num_epochs> -val_split <val_split>

Args:<br>
-bag, -b = Path to the directory containing "<bag_dir>_synced.h5" and "<bag_dir>_vicreg.h5"<br>
-batch_size, -batch = The default value is 256.<br>
-epochs = The default value is 50.<br>
-val_split = Specifies the fraction of the dataset (between 0.0 and 1.0) to reserve for a validation dataset, with the remainder used for the training dataset. The default value is 0.2<br>
'''

## Clustering and creating labeled dataset: cluster_ui.py

'''<br>
python3 cluster_ui.py

Selected vicreg dataset file, synced dataset file, and STERLING trained models file<br>
'''

## Train PATERN pre-adaptation: train_patern_minus.py

'''<br>
python3 train_patern_minus.py -b <bag_dir> -batch_size <batch_size> -epochs <num_epochs> -val_split <val_split>

Args:<br>
-bag, -b = Path to the directory containing "<bag_dir>_synced.h5" and "<bag_dir>_vicreg.h5"<br>
-batch_size, -batch = The default value is 256.<br>
-epochs = The default value is 50.<br>
-val_split = Specifies the fraction of the dataset (between 0.0 and 1.0) to reserve for a validation dataset, with the remainder used for the training dataset. The default value is 0.2<br>
'''

## Train PATERN adaptation: train_patern_plus.py
1. The pre-adaptation data (-pb) is the same data used in STERLING and PATERN pre-adaptation.
2. The adaptation data (-ab) is the new data for training models on new terrains.
3. It will detect terrain data in the adaptation bag that are not within a distance threshold of any clusters from pre-adaptation data. GUI will pop up to show the user patches from adaptation data with extrapolated preference. The user manually assigns the terrain label to new terrain and has the option to manually assign a preference. If the terrain label matches any existing terrain label, it will automatically assign the preference of that existing terrain.

'''<br>
python3 train_patern_plus.py -pb <pb_bag_dir> -ab <ab_bag_dir> -batch_size <batch_size> -epochs <num_epochs> -val_split <val_split>

Args:<br>
-preadapt_bag, -pb = Absolute path to the directory containing "<pb_bag_dir>_synced.h5" and "<pb_bag_dir>_vicreg.h5"<br>
-adapt_bag, -ab = Absolute path to the directory containing "<ab_bag_dir>_synced.h5" and "<ab_bag_dir>_vicreg.h5"<br>
-batch_size, -batch = The default value is 2048.<br>
-epochs = The default value is 50.<br>
-val_split = Specifies the fraction of the dataset (between 0.0 and 1.0) to reserve for a validation dataset, with the remainder used for the training dataset. The default value is 0.2<br>
'''