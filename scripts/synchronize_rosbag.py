"""
Unified processor for multiple ROS1 or ROS2 bags.
- Converts ROS1 → ROS2 if needed (multiple bags supported)
- Supports multiple existing ROS2 bag directories
- Handles timestamp continuity across bags with offsets
- Outputs one synchronized HDF5 + optional video
"""

import argparse
import os
import h5py
import shutil
import time
from pathlib import Path
import gc

import cv2
from cv_bridge import CvBridge
import numpy as np
from rosbags.convert import convert
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CompressedImage, Imu, CameraInfo
from nav_msgs.msg import Odometry
from termcolor import cprint
from tqdm import tqdm
from collections import deque


class SynchronizeRosbag:
    def __init__(self, bag_path, visual, simulation, time_threshold, save_interval=0, skip_first_n=0, skip_last_n=0, save_path=None, image_topic="/panther/oak/rgb/image_raw/compressed"):
        self.BAG_PATH = os.path.normpath(bag_path)
        self.SAVE_PATH = save_path if save_path else self.BAG_PATH
        self.VISUAL = visual
        self.SIM = simulation
        self.TIME_THRESHOLD = time_threshold
        self.save_interval = save_interval
        self.skip_first_n = skip_first_n
        self.skip_last_n = skip_last_n
        self.image_topic = image_topic

        if self.SIM:
            self.odometry_topic = "/odom"
            self.imu_topic = "/imu"
        else:
            self.odometry_topic = "/panther/odometry/wheels"
            self.imu_topic = "/panther/imu/data"

        self.br = CvBridge()
        self.image_msgs = deque()
        self.imu_msgs = deque()
        self.odom_msgs = deque()
        self.synced_msgs = {"image": [], "imu": [], "odom": []}
        self.camera_info = None
        self.global_counter = 0  # For continuous indexing in HDF5

    def _find_ros2_bag_dirs(self):
        """Find all subdirectories that look like valid ROS2 bags."""
        if not os.path.isdir(self.BAG_PATH):
            raise NotADirectoryError(f"{self.BAG_PATH} is not a directory")

        # Check if the path itself is a valid ROS2 bag directory
        yaml_files = [f for f in os.listdir(self.BAG_PATH) if f.endswith("metadata.yaml")]
        db3_files = [f for f in os.listdir(self.BAG_PATH) if f.endswith(".db3")]
        if len(yaml_files) == 1 and len(db3_files) >= 1:
            return [self.BAG_PATH]  # Treat the path itself as the bag dir

        # Look for subdirectories
        candidates = []
        for entry in os.listdir(self.BAG_PATH):
            full_path = os.path.join(self.BAG_PATH, entry)
            if not os.path.isdir(full_path):
                continue
            yaml_files = [f for f in os.listdir(full_path) if f.endswith("metadata.yaml")]
            db3_files = [f for f in os.listdir(full_path) if f.endswith(".db3")]
            if len(yaml_files) == 1 and len(db3_files) >= 1:
                candidates.append(full_path)

        return candidates

    def _get_bag_timestamp_range(self, bag_dir):
        """Get min/max timestamp from relevant topics in a ROS2 bag."""
        db3_files = [f for f in os.listdir(bag_dir) if f.endswith(".db3")]
        if not db3_files:
            return float('inf'), float('-inf')

        db3_path = os.path.join(bag_dir, db3_files[0])
        storage_options = rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions("", "")

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topics = [self.image_topic, self.odometry_topic, self.imu_topic]
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        valid_topics = [t for t in topics if t in topic_types]

        min_time = float('inf')
        max_time = float('-inf')

        while reader.has_next():
            topic, _, t = reader.read_next()
            if topic in valid_topics:
                # Handle both old (Time msg) and new (int) timestamp formats
                if hasattr(t, 'nanoseconds'):
                    ts = t.nanoseconds * 1e-9
                else:
                    ts = t * 1e-9  # t is already in nanoseconds as int
                min_time = min(min_time, ts)
                max_time = max(max_time, ts)

        return min_time, max_time

    def _check_for_ros1_bags(self):
        bag_files = [os.path.join(self.BAG_PATH, f) for f in os.listdir(self.BAG_PATH) if f.endswith('.bag')]
        return [f for f in bag_files if os.path.isfile(f)]

    def _convert_ros1_to_ros2(self, ros1_bags):
        """Convert multiple ROS1 bags into separate ROS2 bag directories."""
        converted_dirs = []
        temp_base = os.path.join(self.BAG_PATH, "ros2_converted")
        os.makedirs(temp_base, exist_ok=True)

        for i, ros1_bag in enumerate(ros1_bags):
            bag_name = Path(ros1_bag).stem
            ros2_dir = os.path.join(temp_base, f"{bag_name}_ros2")
            if os.path.exists(ros2_dir):
                shutil.rmtree(ros2_dir)

            cprint(f"Converting {ros1_bag} → {ros2_dir}", "yellow")
            convert([Path(ros1_bag)], Path(ros2_dir), 'sqlite3', 2)

            # Move to root level with clean name
            final_dir = os.path.join(self.BAG_PATH, bag_name)
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            shutil.move(ros2_dir, final_dir)
            converted_dirs.append(final_dir)

        shutil.rmtree(temp_base)
        return converted_dirs

    def prepare_bags(self):
        """Detect ROS1 or ROS2 bags, convert if needed, return sorted list of ROS2 bag directories."""
        ros1_bags = self._check_for_ros1_bags()
        ros2_dirs = self._find_ros2_bag_dirs()

        if ros1_bags:
            cprint(f"Found {len(ros1_bags)} ROS1 bag(s). Converting to ROS2...", "yellow")
            ros2_dirs = self._convert_ros1_to_ros2(ros1_bags)

        if not ros2_dirs:
            raise FileNotFoundError("No valid ROS2 bags (or convertible ROS1 bags) found.")

        # Get timestamp ranges and sort
        bag_info = []
        for bag_dir in ros2_dirs:
            min_t, max_t = self._get_bag_timestamp_range(bag_dir)
            if min_t == float('inf'):
                cprint(f"Warning: No relevant messages in {bag_dir}", "yellow")
                continue
            bag_info.append((bag_dir, min_t, max_t))

        bag_info.sort(key=lambda x: x[1])
        sorted_dirs = [info[0] for info in bag_info]

        # Calculate offsets for continuity
        self.bag_offsets = []
        max_time_so_far = 0.0
        for _, min_t, max_t in bag_info:
            if min_t <= max_time_so_far:
                offset = max_time_so_far - min_t + 0.01  # small gap
                cprint(f"Applying offset {offset:.3f}s to {os.path.basename(_)}", "red")
            else:
                offset = 0.0
            self.bag_offsets.append(offset)
            max_time_so_far = max(max_time_so_far, max_t + offset)

        cprint(f"Processing {len(sorted_dirs)} ROS2 bag(s) in order:", "green")
        for d in sorted_dirs:
            cprint(f"  → {os.path.basename(d)}", "cyan")

        return sorted_dirs

    def image_callback(self, msg, timestamp_offset=0.0):
        if isinstance(msg, Image):
            cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")
            compressed_msg = self.br.cv2_to_compressed_imgmsg(cv_image)
            compressed_msg.header = msg.header
            self.image_msgs.append((compressed_msg, timestamp_offset))
        elif isinstance(msg, CompressedImage):
            self.image_msgs.append((msg, timestamp_offset))
        self.sync_messages()

    def imu_callback(self, msg, timestamp_offset=0.0):
        self.imu_msgs.append((msg, timestamp_offset))
        self.sync_messages()

    def odom_callback(self, msg, timestamp_offset=0.0):
        self.odom_msgs.append((msg, timestamp_offset))
        self.sync_messages()

    def sync_messages(self):
        while self.image_msgs and self.imu_msgs and self.odom_msgs:
            (img_msg, img_off) = self.image_msgs[0]
            (imu_msg, imu_off) = self.imu_msgs[0]
            (odom_msg, odom_off) = self.odom_msgs[0]

            image_time = (img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9) + img_off
            imu_time = (imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9) + imu_off
            odom_time = (odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9) + odom_off

            avg_time = (image_time + imu_time + odom_time) / 3.0
            if all(abs(t - avg_time) < self.TIME_THRESHOLD for t in [image_time, imu_time, odom_time]):
                self.image_msgs.popleft()
                self.imu_msgs.popleft()
                self.odom_msgs.popleft()

                img_data = np.frombuffer(img_msg.data, np.uint8)
                self.synced_msgs["image"].append({"timestamp": image_time, "data": img_data})
                self.synced_msgs["imu"].append({
                    "timestamp": imu_time,
                    "orientation": np.array([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w]),
                    "angular_velocity": np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z]),
                    "linear_acceleration": np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z]),
                })
                self.synced_msgs["odom"].append({
                    "timestamp": odom_time,
                    "pose": np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z,
                                     odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
                                     odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]),
                    "twist": np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z,
                                      odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z]),
                })
            else:
                times = [image_time, imu_time, odom_time]
                if min(times) == image_time:
                    self.image_msgs.popleft()
                elif min(times) == imu_time:
                    self.imu_msgs.popleft()
                else:
                    self.odom_msgs.popleft()

    def process_single_bag(self, bag_dir, timestamp_offset):
        db3_files = [f for f in os.listdir(bag_dir) if f.endswith(".db3")]
        db3_path = os.path.join(bag_dir, db3_files[0])

        storage_options = rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions("", "")
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

        total_msgs = reader.get_metadata().message_count
        cprint(f"Processing {os.path.basename(bag_dir)} ({total_msgs} messages)", "yellow")

        with tqdm(total=total_msgs, desc=f"Reading {os.path.basename(bag_dir)}") as pbar:
            while reader.has_next():
                topic, data, _ = reader.read_next()
                msg_type = topic_types.get(topic)

                if msg_type in ["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"] and topic == self.image_topic:
                    cls = Image if msg_type == "sensor_msgs/msg/Image" else CompressedImage
                    msg = deserialize_message(data, cls)
                    self.image_callback(msg, timestamp_offset)
                elif msg_type == "sensor_msgs/msg/Imu" and topic == self.imu_topic:
                    msg = deserialize_message(data, Imu)
                    self.imu_callback(msg, timestamp_offset)
                elif msg_type == "nav_msgs/msg/Odometry" and topic == self.odometry_topic:
                    msg = deserialize_message(data, Odometry)
                    self.odom_callback(msg, timestamp_offset)
                elif msg_type == "sensor_msgs/msg/CameraInfo" and self.camera_info is None:
                    msg = deserialize_message(data, CameraInfo)
                    self.camera_info = msg

                pbar.update(1)

    def read_rosbag(self):
        bag_dirs = self.prepare_bags()

        # Clear buffers
        self.image_msgs.clear()
        self.imu_msgs.clear()
        self.odom_msgs.clear()
        self.synced_msgs = {"image": [], "imu": [], "odom": []}

        # Process each bag with its offset
        for bag_dir, offset in zip(bag_dirs, self.bag_offsets):
            self.process_single_bag(bag_dir, offset)

            # Save batch
            self.save_data(batch_mode=True)
            cprint(f"Saved batch from {os.path.basename(bag_dir)} ({len(self.synced_msgs['image'])} new triplets)", "green")

            # Clear synced_msgs for next bag (but keep global_counter)
            self.synced_msgs = {"image": [], "imu": [], "odom": []}
            gc.collect()
        pass

    def save_data(self, skip_last_n=0, skip_first_n=0, batch_mode=False):
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        file_path = os.path.join(self.SAVE_PATH, os.path.basename(self.BAG_PATH) + "_synced.h5")

        if batch_mode:
            if not self.synced_msgs["image"]:
                return

            with h5py.File(file_path, 'a') as h5f:
                img_grp = h5f.require_group('image')
                imu_grp = h5f.require_group('imu')
                odom_grp = h5f.require_group('odom')

                for msg in self.synced_msgs["image"]:
                    sub = img_grp.create_group(str(self.global_counter))
                    sub.create_dataset('timestamp', data=msg["timestamp"])
                    sub.create_dataset('data', data=msg["data"])
                    self.global_counter += 1

                for i, msg in enumerate(self.synced_msgs["imu"]):
                    sub = imu_grp.create_group(str(self.global_counter - len(self.synced_msgs["image"]) + i))
                    sub.create_dataset('timestamp', data=msg["timestamp"])
                    sub.create_dataset('orientation', data=msg["orientation"])
                    sub.create_dataset('angular_velocity', data=msg["angular_velocity"])
                    sub.create_dataset('linear_acceleration', data=msg["linear_acceleration"])

                for i, msg in enumerate(self.synced_msgs["odom"]):
                    sub = odom_grp.create_group(str(self.global_counter - len(self.synced_msgs["image"]) + i))
                    sub.create_dataset('timestamp', data=msg["timestamp"])
                    sub.create_dataset('pose', data=msg["pose"])
                    sub.create_dataset('twist', data=msg["twist"])

            return

        # Final mode: video + filtered HDF5
        if self.VISUAL and self.camera_info:
            frame_size = (self.camera_info.width, self.camera_info.height)
            video_path = os.path.join(self.SAVE_PATH, "original.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, frame_size)

            with h5py.File(file_path, 'r') as h5f:
                n = len(h5f['image'])
                indices = [i for i in range(skip_first_n, n - skip_last_n) if self.save_interval == 0 or (i % self.save_interval == 0)]
                for i in tqdm(indices, desc="Writing video"):
                    data = h5f['image'][str(i)]['data'][()]
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        writer.write(img)
            writer.release()
            cprint(f"Video saved: {video_path}", "green")

        # Optional: rewrite HDF5 with filtered indices (if save_interval or skip_first/last used)
        if self.save_interval > 0 or skip_last_n > 0 or skip_first_n > 0:
            temp_path = file_path + ".tmp"
            with h5py.File(file_path, 'r') as src, h5py.File(temp_path, 'w') as dst:
                for group_name in ['image', 'imu', 'odom']:
                    src_g = src[group_name]
                    dst_g = dst.create_group(group_name)
                    n = len(src_g)
                    indices = [i for i in range(skip_first_n, n - skip_last_n) if self.save_interval == 0 or (i % self.save_interval == 0)]
                    for new_idx, old_idx in enumerate(indices):
                        old_sub = src_g[str(old_idx)]
                        new_sub = dst_g.create_group(str(new_idx))
                        for ds_name in old_sub.keys():
                            new_sub.create_dataset(ds_name, data=old_sub[ds_name][()])
            os.replace(temp_path, file_path)

        cprint(f"Final HDF5 saved: {file_path} ({self.global_counter} total triplets)", "green")

    def calculate_avg_inter_group_time_difference(self):
        file_path = os.path.join(self.SAVE_PATH, os.path.basename(self.BAG_PATH) + "_synced.h5")
        if not os.path.exists(file_path):
            return

        with h5py.File(file_path, 'r') as h5f:
            n = len(h5f['image'])
            if n < 2:
                cprint("Not enough triplets for timing analysis.", "yellow")
                return

            ts = [h5f['image'][str(i)]['timestamp'][()] for i in range(n)]
            diffs = np.diff(ts)

            print(f"\nTiming Analysis ({n} synchronized triplets):")
            print(f"Total duration: {ts[-1] - ts[0]:.3f}s")
            print(f"Average interval: {np.mean(diffs):.6f}s")
            print(f"Min interval: {np.min(diffs):.6f}s")
            print(f"Max interval: {np.max(diffs):.6f}s")
            print(f"Std dev: {np.std(diffs):.6f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple ROS1/ROS2 bags → synchronized HDF5")
    parser.add_argument("--bag_path", "-b", type=str, required=True, help="Directory containing ROS bag(s)")
    parser.add_argument("--save_path", "-s", type=str, default=None, help="Directory to save output")
    parser.add_argument("--visual", "-v", action="store_true", help="Generate video")
    parser.add_argument("--simulation", "-sim", action="store_true", help="Use simulation topics")
    parser.add_argument("--threshold", "-th", type=float, default=0.05, help="Sync time threshold (s)")
    parser.add_argument("--skip_last_n", "-n", type=int, default=0, help="Skip last N triplets")
    parser.add_argument("--skip_first_n", "-f", type=int, default=0, help="Skip first N triplets")
    parser.add_argument("--save_interval", "-int", type=int, default=0, help="Save every Nth triplet (0 = all)")
    parser.add_argument("--image_topic", "-it", type=str, default="/panther/oak/rgb/image_raw/compressed", help="Image topic name")
    args = parser.parse_args()

    processor = SynchronizeRosbag(
        bag_path=args.bag_path,
        visual=args.visual,
        simulation=args.simulation,
        time_threshold=args.threshold,
        save_interval=args.save_interval,
        skip_first_n=args.skip_first_n,
        skip_last_n=args.skip_last_n,
        save_path=args.save_path,
        image_topic=args.image_topic
    )
    processor.read_rosbag()
    processor.save_data(
        skip_first_n=processor.skip_first_n,
        skip_last_n=processor.skip_last_n,
        batch_mode=False
    )
    processor.calculate_avg_inter_group_time_difference()
