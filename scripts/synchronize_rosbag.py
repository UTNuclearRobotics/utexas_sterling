"""
This file processes a rosbag converting camera, IMU, and odometry data into a
Python dictionary and saves it as a HDF5 file.

The main functionalities include:
1. Convert ROS1 .bag to ROS2 format if necessary using rosbags.convert
2. Read the rosbag sequentially using ROS2 deserialization
3. Synchronize the received data
4. Process the data to extract patches from images based on odometry information
5. Save the processed data into an HDF5 file
"""

import argparse
import os
import h5py
import shutil
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
from rosbags.convert import convert
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, Imu
from termcolor import cprint
from tqdm import tqdm
from collections import deque


class SynchronizeRosbag:
    """
    Class to process a ROS2 bag file or convert ROS1 bag file to ROS2 bag files and extract patches from images
    based on odometry information.
    """

    def __init__(self, bag_path, visual, simulation, time_threshold):
        self.BAG_PATH = bag_path
        self.SAVE_PATH = bag_path
        self.VISUAL = visual
        self.SIM = simulation
        self.TIME_THRESHOLD = time_threshold

        if self.SIM:
            self.odometry_topic = "/odom"
            self.imu_topic = "/imu"
        else:
            self.odometry_topic = "/odom"
            self.imu_topic = "/imu"

        # Check if a .bag file exists within bag_path
        self.is_ros1, self.ros1_bag_file = self._check_for_ros1_bag(bag_path)
        self.convert_path = self._check_or_convert_to_ros2(bag_path)
        self.SAVE_PATH = self.BAG_PATH  # Save in the same directory as the bag
        
        self.br = CvBridge()
        self.image_msgs = deque()
        self.imu_msgs = deque()
        self.odom_msgs = deque()
        self.synced_msgs = {"image": [], "imu": [], "odom": []}

    def _check_for_ros1_bag(self, bag_path):
        """Check if there is a .bag file within bag_path."""
        if not os.path.exists(bag_path):
            raise FileNotFoundError(f"Path does not exist: bag_path:={bag_path}")
        
        if not os.path.isdir(bag_path):
            raise ValueError(f"bag_path must be a directory: {bag_path}")
        
        bag_files = [f for f in os.listdir(bag_path) if f.endswith('.bag')]
        if len(bag_files) > 1:
            raise ValueError(f"Multiple .bag files found in {bag_path}. Please specify a single ROS1 bag.")
        elif len(bag_files) == 1:
            return True, os.path.join(bag_path, bag_files[0])
        return False, None

    def _check_or_convert_to_ros2(self, bag_path):
        """Check if ROS2 bag exists in bag_path; convert ROS1 to ROS2 if needed."""
        yaml_files = [f for f in os.listdir(bag_path) if f.endswith("metadata.yaml")]
        db3_files = [f for f in os.listdir(bag_path) if f.endswith(".db3")]

        # If ROS2 files exist in the root directory, use them
        if len(yaml_files) == 1 and len(db3_files) >= 1:
            cprint(f"Found existing ROS2 bag at {bag_path}, using it", "yellow")
            return bag_path

        # If ROS1 bag exists and no valid ROS2 files in root, convert it
        if self.is_ros1:
            # Use a temporary subdirectory for conversion
            ros2_bag_path = os.path.join(bag_path, "ros2_converted")
            cprint(f"Checking conversion directory: {ros2_bag_path}", "yellow")

            # Ensure the directory is clean
            if os.path.exists(ros2_bag_path):
                cprint(f"Removing existing directory: {ros2_bag_path}", "yellow")
                shutil.rmtree(ros2_bag_path)
                time.sleep(0.1)  # Small delay to ensure filesystem sync

            cprint(f"Creating new directory: {ros2_bag_path}", "yellow")

            cprint(f"Converting ROS1 bag {self.ros1_bag_file} to ROS2 format at {ros2_bag_path}", "yellow")
            try:
                convert(
                    [Path(self.ros1_bag_file)],  # srcs: Sequence of source ROS1 bags
                    Path(ros2_bag_path),         # dst: Destination ROS2 bag directory
                    'sqlite3',                   # dst_storage: Use SQLite3 for ROS2
                    2,                           # dst_version: ROS2 version
                    None,                        # compress: No compression
                    'file',                      # compress_mode: Default mode
                    None,                        # default_typestore: No specific default
                    None,                        # typestore: No specific destination typestore
                    [],                          # exclude_topics: No topics excluded
                    [],                          # include_topics: Include all topics
                    [],                          # exclude_msgtypes: No message types excluded
                    []                           # include_msgtypes: Include all message types
                )
                # Check the contents of the converted directory
                converted_files = os.listdir(ros2_bag_path)
                cprint(f"Conversion completed. Contents of {ros2_bag_path}: {converted_files}", "green")
                
                # Verify ROS2 bag structure
                yaml_files = [f for f in converted_files if f.endswith("metadata.yaml")]
                db3_files = [f for f in converted_files if f.endswith(".db3")]
                if not yaml_files or not db3_files:
                    raise FileNotFoundError(
                        f"Conversion failed to produce a valid ROS2 bag in {ros2_bag_path}. "
                        f"Expected metadata.yaml and .db3 files, found: {converted_files}"
                    )

                # Move files to bag_path and rename .db3 file
                bag_name = os.path.basename(bag_path)  # e.g., "2025-04-10-18-15-09"
                target_db3_name = f"{bag_name}.db3"
                cprint(f"Moving and renaming files to {bag_path}", "yellow")

                # Move metadata.yaml
                if len(yaml_files) == 1:
                    src_yaml = os.path.join(ros2_bag_path, yaml_files[0])
                    dst_yaml = os.path.join(bag_path, "metadata.yaml")
                    shutil.move(src_yaml, dst_yaml)
                    cprint(f"Moved {src_yaml} to {dst_yaml}", "green")

                # Move and rename the first .db3 file
                if db3_files:
                    src_db3 = os.path.join(ros2_bag_path, db3_files[0])
                    dst_db3 = os.path.join(bag_path, target_db3_name)
                    shutil.move(src_db3, dst_db3)
                    cprint(f"Moved and renamed {src_db3} to {dst_db3}", "green")

                    # Handle additional .db3 files if present (though rare with sqlite3)
                    for extra_db3 in db3_files[1:]:
                        src_extra = os.path.join(ros2_bag_path, extra_db3)
                        dst_extra = os.path.join(bag_path, extra_db3)
                        shutil.move(src_extra, dst_extra)
                        cprint(f"Moved {src_extra} to {dst_extra}", "green")

                # Remove the now-empty ros2_converted directory
                cprint(f"Removing temporary directory: {ros2_bag_path}", "yellow")
                shutil.rmtree(ros2_bag_path)
                time.sleep(0.1)  # Ensure filesystem sync

                cprint(f"Final contents of {bag_path}: {os.listdir(bag_path)}", "green")
                return bag_path  # Return the parent directory with the moved files
            except Exception as e:
                cprint(f"Error during conversion or file handling: {str(e)}", "red")
                if os.path.exists(ros2_bag_path):
                    cprint(f"Cleaning up failed conversion directory: {ros2_bag_path}", "yellow")
                    shutil.rmtree(ros2_bag_path)
                raise

        raise FileNotFoundError(f"No valid ROS bag found in {bag_path}. Expected a .bag file or ROS2 bag files.")

    def image_callback(self, msg):
        if isinstance(msg, Image):
            if self.SIM:
                # Convert raw image to compressed
                cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                compressed_msg = self.br.cv2_to_compressed_imgmsg(cv_image)
                compressed_msg.header = msg.header
                self.image_msgs.append(compressed_msg)
            else:
                # Store raw image (or convert to compressed if needed)
                cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                compressed_msg = self.br.cv2_to_compressed_imgmsg(cv_image)
                compressed_msg.header = msg.header
                self.image_msgs.append(compressed_msg)
        elif isinstance(msg, CompressedImage):
            self.image_msgs.append(msg)
        else:
            cprint(f"Unsupported image message type: {type(msg)}", "red")
            return
        self.sync_messages()

    def imu_callback(self, msg):
        self.imu_msgs.append(msg)
        self.sync_messages()

    def odom_callback(self, msg):
        self.odom_msgs.append(msg)
        self.sync_messages()

    def sync_messages(self):
        while self.image_msgs and self.imu_msgs and self.odom_msgs:
            image_time = self.image_msgs[0].header.stamp.sec + self.image_msgs[0].header.stamp.nanosec * 1e-9
            imu_time = self.imu_msgs[0].header.stamp.sec + self.imu_msgs[0].header.stamp.nanosec * 1e-9
            odom_time = self.odom_msgs[0].header.stamp.sec + self.odom_msgs[0].header.stamp.nanosec * 1e-9

            # Find the average timestamp
            avg_time = (image_time + imu_time + odom_time) / 3.0

            # Calculate time differences
            time_diff_image = abs(image_time - avg_time)
            time_diff_imu = abs(imu_time - avg_time)
            time_diff_odom = abs(odom_time - avg_time)

            # Synchronize if all time differences are within the threshold
            if time_diff_image < self.TIME_THRESHOLD and time_diff_imu < self.TIME_THRESHOLD and time_diff_odom < self.TIME_THRESHOLD:
                img_msg = self.image_msgs.popleft()
                imu_msg = self.imu_msgs.popleft()
                odom_msg = self.odom_msgs.popleft()

                # Process image message
                img_data = np.frombuffer(img_msg.data, np.uint8)
                # img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                img_msg_fields = {"timestamp": image_time, "data": img_data}
                self.synced_msgs["image"].append(img_msg_fields)

                # Process IMU message
                imu_msg_fields = {
                    "timestamp": imu_time,
                    "orientation": np.array(
                        [imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w]
                    ),
                    "angular_velocity": np.array(
                        [imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z]
                    ),
                    "linear_acceleration": np.array(
                        [imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z]
                    ),
                }
                self.synced_msgs["imu"].append(imu_msg_fields)

                # Process odometry message
                odom_msg_fields = {
                    "timestamp": odom_time,
                    "pose": np.array(
                        [
                            odom_msg.pose.pose.position.x,
                            odom_msg.pose.pose.position.y,
                            odom_msg.pose.pose.position.z,
                            odom_msg.pose.pose.orientation.x,
                            odom_msg.pose.pose.orientation.y,
                            odom_msg.pose.pose.orientation.z,
                            odom_msg.pose.pose.orientation.w,
                        ]
                    ),
                    "twist": np.array(
                        [
                            odom_msg.twist.twist.linear.x,
                            odom_msg.twist.twist.linear.y,
                            odom_msg.twist.twist.linear.z,
                            odom_msg.twist.twist.angular.x,
                            odom_msg.twist.twist.angular.y,
                            odom_msg.twist.twist.angular.z,
                        ]
                    ),
                }
                self.synced_msgs["odom"].append(odom_msg_fields)
            else:
                # Discard the earliest message to find a better match
                if image_time <= imu_time and image_time <= odom_time:
                    self.image_msgs.popleft()
                elif imu_time <= image_time and imu_time <= odom_time:
                    self.imu_msgs.popleft()
                else:
                    self.odom_msgs.popleft()

    def read_rosbag(self):
        """
        Reads and processes messages from a ROS2 bag file.
        """
        if not os.path.exists(self.BAG_PATH):
            raise FileNotFoundError(f"Path does not exist: bag_path:={self.BAG_PATH}")

        # Validate the path is a rosbag by checking for metadata.yaml and .db3 file
        yaml_files = [file for file in os.listdir(self.BAG_PATH) if file.endswith("metadata.yaml")]
        db3_files = [file for file in os.listdir(self.BAG_PATH) if file.endswith(".db3")]
        if len(yaml_files) != 1 or len(db3_files) != 1:
            raise FileNotFoundError(
                f"Invalid bag. Bag must contain exactly one .yaml and one .db3 file: {self.BAG_PATH}"
            )

        # Print the files found for debugging
        cprint(f"Found metadata: {yaml_files}", "yellow")
        cprint(f"Found db3: {db3_files}", "yellow")

        # Use the full path to the .db3 file instead of the directory
        db3_path = os.path.join(self.BAG_PATH, db3_files[0])
        storage_options = rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3")
        converter_options = rosbag2_py._storage.ConverterOptions("", "")
        reader = rosbag2_py.SequentialReader()
        
        cprint(f"Opening ROS2 bag at: {db3_path}", "yellow")
        reader.open(storage_options, converter_options)

        # Print rosbag metadata
        metadata = reader.get_metadata()
        print(f"{metadata}")

        # Iterator variables
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        # Iterate through the messages in the rosbag
        with tqdm(total=metadata.message_count, desc="Reading rosbag messages") as pbar:
            while reader.has_next():
                (topic, msg, t) = reader.read_next()
                topic_type = type_map.get(topic)

                match topic_type:
                    case "sensor_msgs/msg/Image":
                        msg = deserialize_message(msg, Image)
                        self.image_callback(msg)
                    case "sensor_msgs/msg/CompressedImage":
                        msg = deserialize_message(msg, CompressedImage)
                        self.image_callback(msg)
                    case "sensor_msgs/msg/CameraInfo":
                        msg = deserialize_message(msg, CameraInfo)
                        self.camera_info = msg
                    case "nav_msgs/msg/Odometry":
                        if topic == self.odometry_topic:
                            msg = deserialize_message(msg, Odometry)
                            self.odom_callback(msg)
                    case "sensor_msgs/msg/Imu":
                        if topic == self.imu_topic:
                            msg = deserialize_message(msg, Imu)
                            self.imu_callback(msg)

                pbar.update(1)

    def save_data(self):
        if self.VISUAL:
            # Video writing code remains unchanged
            frame_size = (self.camera_info.width, self.camera_info.height)
            #frame_size = (1080, 1920)
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_save_path = os.path.join(self.BAG_PATH, "original.mp4")
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, frame_size)

            for i in tqdm(range(len(self.synced_msgs["image"])), desc="Writing video"):
                img_data = self.synced_msgs["image"][i]["data"]
                cprint(f"Decoding image {i} with {len(img_data)} bytes", "blue")
                img = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)

                if img is None:
                    cprint(f"Error: Failed to decode image {i}", "red")
                    continue

                cprint(f"Decoded image {i} with shape {img.shape}", "green")
                video_writer.write(img)

            video_writer.release()
            cprint(f"Video saved successfully: {video_save_path}", "green")

        # Save the data as an HDF5 file instead of pickle
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        file_path = os.path.join(self.SAVE_PATH, self.BAG_PATH.split("/")[-1] + "_synced.h5")
        
        with h5py.File(file_path, 'w') as h5f:
            # Create groups for each message type
            image_group = h5f.create_group('image')
            imu_group = h5f.create_group('imu')
            odom_group = h5f.create_group('odom')

            # Save image data
            for i, msg in enumerate(self.synced_msgs['image']):
                subgroup = image_group.create_group(str(i))
                subgroup.create_dataset('timestamp', data=msg['timestamp'])
                subgroup.create_dataset('data', data=msg['data'])

            # Save IMU data
            for i, msg in enumerate(self.synced_msgs['imu']):
                subgroup = imu_group.create_group(str(i))
                subgroup.create_dataset('timestamp', data=msg['timestamp'])
                subgroup.create_dataset('orientation', data=msg['orientation'])
                subgroup.create_dataset('angular_velocity', data=msg['angular_velocity'])
                subgroup.create_dataset('linear_acceleration', data=msg['linear_acceleration'])

            # Save odometry data
            for i, msg in enumerate(self.synced_msgs['odom']):
                subgroup = odom_group.create_group(str(i))
                subgroup.create_dataset('timestamp', data=msg['timestamp'])
                subgroup.create_dataset('pose', data=msg['pose'])
                subgroup.create_dataset('twist', data=msg['twist'])

        cprint(f"Data saved successfully as HDF5: {file_path}", "green")
        cprint(f"Total synced messages: {len(self.synced_msgs['imu'])}", "green")

    def calculate_avg_inter_group_time_difference(self):
        """
        Calculate the average time difference between consecutive synchronized triplets
        and the total recording time of the synchronized messages.
        """
        if len(self.synced_msgs["image"]) < 2:
            print("Need at least 2 synchronized triplets to calculate inter-group time differences.")
            return

        num_triplets = len(self.synced_msgs["image"])
        print(f"\nAnalyzing inter-group time differences for {num_triplets} synchronized triplets:")
        print(f"Time threshold used for synchronization: {self.TIME_THRESHOLD} seconds")

        # Calculate total recording time (using image timestamps as reference)
        first_image_time = self.synced_msgs["image"][0]["timestamp"]
        last_image_time = self.synced_msgs["image"][-1]["timestamp"]
        total_recording_time = last_image_time - first_image_time

        # Lists to store inter-group differences for each message type
        diffs_image = []
        diffs_imu = []
        diffs_odom = []

        # Calculate differences between consecutive triplets
        print(f"{'Index':<6} {'Image Diff (s)':<15} {'IMU Diff (s)':<15} {'Odom Diff (s)':<15}")
        for i in range(num_triplets - 1):
            image_diff = self.synced_msgs["image"][i + 1]["timestamp"] - self.synced_msgs["image"][i]["timestamp"]
            imu_diff = self.synced_msgs["imu"][i + 1]["timestamp"] - self.synced_msgs["imu"][i]["timestamp"]
            odom_diff = self.synced_msgs["odom"][i + 1]["timestamp"] - self.synced_msgs["odom"][i]["timestamp"]

            diffs_image.append(image_diff)
            diffs_imu.append(imu_diff)
            diffs_odom.append(odom_diff)

            print(f"{i:<6} {image_diff:<15.6f} {imu_diff:<15.6f} {odom_diff:<15.6f}")

        # Compute averages
        avg_diff_image = np.mean(diffs_image)
        avg_diff_imu = np.mean(diffs_imu)
        avg_diff_odom = np.mean(diffs_odom)
        overall_avg_diff = np.mean([avg_diff_image, avg_diff_imu, avg_diff_odom])

        # Display results
        print("\nTotal Recording Time (based on synchronized image messages):")
        print(f"Total time: {total_recording_time:.6f} seconds")
        print(f"First timestamp: {first_image_time:.6f} seconds")
        print(f"Last timestamp: {last_image_time:.6f} seconds")

        print("\nAverage Inter-Group Time Differences:")
        print(f"Image: {avg_diff_image:.6f} seconds")
        print(f"IMU: {avg_diff_imu:.6f} seconds")
        print(f"Odom: {avg_diff_odom:.6f} seconds")
        print(f"Overall Average (across all types): {overall_avg_diff:.6f} seconds")

        # Relate total time to number of triplets
        expected_avg_diff = total_recording_time / (num_triplets - 1) if num_triplets > 1 else 0
        print(f"\nExpected average difference (total time / (num_triplets - 1)): {expected_avg_diff:.6f} seconds")

        # Additional statistics (image-based)
        print("\nAdditional Statistics (Image-based):")
        print(f"Min difference: {min(diffs_image):.6f} seconds")
        print(f"Max difference: {max(diffs_image):.6f} seconds")
        print(f"Std deviation: {np.std(diffs_image):.6f} seconds")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a ROS2 bag to an HDF5 file.")
    parser.add_argument("--bag_path", "-b", type=str, required=True, help="Path to the ROS2 bag file.")
    parser.add_argument("--save_path", "-s", default=None, type=str,help="Path to save the processed data.",)
    parser.add_argument("--visual", "-v", action="store_true", default=False, help="Save video of processed rosbag.")
    parser.add_argument("--simulation", "-sim", action="store_true", default=False, help="Rosbag is from a Gazebo simulation.")
    parser.add_argument("--threshold", "-th", type=float, default=0.05, help="Threshold for syncing messages within a certain time window")
    args = parser.parse_args()

    cprint(
        f"Bag Path: {args.bag_path}\n"
        f"Save Path: {args.save_path}\n"
        f"Visualization: {args.visual}\n"
        f"Simulation: {args.simulation}\n"
        f"Time Threshold: {args.threshold}",
        "blue",
    )

    processor = SynchronizeRosbag(
        bag_path=os.path.normpath(args.bag_path),
        visual=args.visual,
        simulation=args.simulation,
        time_threshold=args.threshold
    )
    processor.read_rosbag()
    processor.save_data()
    processor.calculate_avg_inter_group_time_difference()
