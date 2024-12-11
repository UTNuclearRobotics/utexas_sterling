"""
This file processes a rosbag converting camera, IMU, and odometry data into a
Python dictionary and saves it as a pickle file.

The main functionalities include:
1. Read the rosbag sequentially.
2. Synchronizing the received data.
3. Processing the data to extract patches from images based on odometry information.
4. Saving the processed data into a pickle file.
"""

import argparse
import os
import pickle

import cv2
import numpy as np
import rosbag2_py
import yaml
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu
from termcolor import cprint
from tqdm import tqdm
from collections import deque


class SynchronizeRosbag:
    """
    Class to process a ROS2 bag file and extract patches from images
    based on odometry information.
    """

    def __init__(self, bag_path, visual):
        self.BAG_PATH = bag_path
        self.SAVE_PATH = bag_path
        self.VISUAL = visual

        # Initialize queues
        self.image_msgs = deque()
        self.imu_msgs = deque()
        self.odom_msgs = deque()

        # Lists to store synchronized messages
        self.synced_msgs = {"image": [], "imu": [], "odom": []}

        self.camera_info = None

    def image_callback(self, msg):
        self.image_msgs.append(msg)
        self.sync_messages()

    def imu_callback(self, msg):
        self.imu_msgs.append(msg)
        self.sync_messages()

    def odom_callback(self, msg):
        self.odom_msgs.append(msg)
        self.sync_messages()

    def sync_messages(self):
        print(f"Image: {len(self.image_msgs)} IMU: {len(self.imu_msgs)} Odom: {len(self.odom_msgs)}")
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
            if time_diff_image < 0.05 and time_diff_imu < 0.05 and time_diff_odom < 0.05:
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
        # Check if the bag file exists
        if not os.path.exists(self.BAG_PATH):
            raise FileNotFoundError(f"Path does not exist: bag_path:={self.BAG_PATH}")

        # Validate the path is a rosbag by checking for metadata.yaml and .db3 file
        yaml_files = [file for file in os.listdir(self.BAG_PATH) if file.endswith(".yaml")]
        db3_files = [file for file in os.listdir(self.BAG_PATH) if file.endswith(".db3")]
        if len(yaml_files) != 1 or len(db3_files) != 1:
            raise FileNotFoundError(
                f"Invalid bag. Bag must contain exactly one .yaml and one .db3 file: {self.BAG_PATH}"
            )

        # Set up storage and converter options for reading the rosbag
        storage_options = rosbag2_py.StorageOptions(uri=self.BAG_PATH, storage_id="sqlite3")
        converter_options = rosbag2_py._storage.ConverterOptions("", "")
        reader = rosbag2_py.SequentialReader()
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
                    case "sensor_msgs/msg/CompressedImage":
                        msg = deserialize_message(msg, CompressedImage)
                        self.image_callback(msg)
                    case "sensor_msgs/msg/CameraInfo":
                        msg = deserialize_message(msg, CameraInfo)
                        self.camera_info = msg
                    case "nav_msgs/msg/Odometry":
                        msg = deserialize_message(msg, Odometry)
                        self.odom_callback(msg)
                    case "sensor_msgs/msg/Imu":
                        msg = deserialize_message(msg, Imu)
                        self.imu_callback(msg)

                pbar.update(1)

    def save_data(self):
        if self.VISUAL:
            # Initialize the video writer
            frame_size = (self.camera_info.width, self.camera_info.height)
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_save_path = os.path.join(self.BAG_PATH, self.BAG_PATH.split("/")[-1] + ".mp4")
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, frame_size)

            for i in tqdm(range(len(self.synced_msgs)), desc="Writing video"):
                img_msg = self.synced_msgs[i][1]
                img = np.frombuffer(img_msg.data, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                video_writer.write(img)

            video_writer.release()
            cprint(f"Video saved successfully: {video_save_path}", "green")

        # Save the data as a pickle file
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        file_path = os.path.join(self.SAVE_PATH, self.BAG_PATH.split("/")[-1] + ".pkl")
        with open(file_path, "wb") as file:
            pickle.dump(self.synced_msgs, file)
        cprint(f"Data saved successfully: {file_path}", "green")
        cprint(f"Total synced messages: {len(self.synced_msgs['imu'])}", "green")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a ROS2 bag to a pickle file.")
    parser.add_argument("--bag_path", "-b", type=str, required=True, help="Path to the ROS2 bag file.")
    parser.add_argument(
        "--save_path",
        "-s",
        default=None,
        type=str,
        help="Path to save the processed data.",
    )
    parser.add_argument("--visual", "-v", action="store_true", default=False, help="Save video of processed rosbag.")
    args = parser.parse_args()

    cprint(
        f"Bag Path: {args.bag_path}\n" f"Save Path: {args.save_path}\n" f"Visualization: {args.visual}",
        "blue",
    )

    # Process the rosbag
    processor = SynchronizeRosbag(
        bag_path=os.path.normpath(args.bag_path),
        visual=args.visual,
    )
    processor.read_rosbag()
    processor.save_data()
