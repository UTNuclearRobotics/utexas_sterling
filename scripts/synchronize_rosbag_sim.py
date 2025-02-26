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
from cv_bridge import CvBridge
import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, Imu
from termcolor import cprint
from tqdm import tqdm
from collections import deque

class ImageStitcher:
    def estimate_homography(keypoints1, keypoints2, matches, threshold=3):
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
        return H, mask
    
    def warp_images(img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, H)

        corners = np.concatenate((corners1, warped_corners2), axis=0)
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        warped_img2 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
        warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

        return warped_img2
    
    def blend_images(img1, img2):
        mask = np.where(img1 != 0, 1, 0).astype(np.float32)
        blended_img = img1 * mask + img2 * (1 - mask)
        return blended_img.astype(np.uint8)

    def stitch_images(images):
        # Detect ORB keypoints and descriptors in the images
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(images[0], None)
        keypoints2, descriptors2 = orb.detectAndCompute(images[1], None)
        keypoints3, descriptors3 = orb.detectAndCompute(images[2], None)

        # Match features between the images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches12 = bf.match(descriptors1, descriptors2)
        matches23 = bf.match(descriptors2, descriptors3)

        # Sort matches by distance
        matches12 = sorted(matches12, key=lambda x: x.distance)
        matches23 = sorted(matches23, key=lambda x: x.distance)

        # Extract location of good matches
        h12, mask12 = ImageStitcher.estimate_homography(keypoints1, keypoints2, matches12)
        h23, mask23 = ImageStitcher.estimate_homography(keypoints2, keypoints3, matches23)

        # Warp images
        warped_img12 = ImageStitcher.warp_images(images[0], images[1], h12)
        warped_img23 = ImageStitcher.warp_images(images[1], images[2], h23)

        # Combine the two panoramas
        return warped_img23
        stitched_img = ImageStitcher.blend_images(warped_img12, images[0])

        return stitched_img

class SynchronizeRosbag:
    """
    Class to process a ROS2 bag file and extract patches from images
    based on odometry information.
    """

    def __init__(self, bag_path, visual):
        self.BAG_PATH = bag_path
        self.SAVE_PATH = bag_path
        self.VISUAL = visual

        self.odometry_topic = "/odometry/filtered"
        self.imu_topic = "/imu/data"
        self.img_topic1 = "/oakd1/oak_d_node/rgb/image_rect_color"
        self.img_topic2 = "/oakd2/oak_d_node/rgb/image_rect_color"
        self.img_topic3 = "/oakd3/oak_d_node/rgb/image_rect_color"

        # Bridge for conversions between Image and CompressedImage
        self.br = CvBridge()

        # Initialize queues
        self.image1_msgs = deque()
        self.image2_msgs = deque()
        self.image3_msgs = deque()
        self.imu_msgs = deque()
        self.odom_msgs = deque()

        # Lists to store synchronized messages
        self.synced_msgs = {"image": [], "imu": [], "odom": []}

        self.camera_info = None

    def image_callback(self, msg, queueNum):
        image = msg
        msg = self.br.imgmsg_to_cv2(image, desired_encoding="bgr8")
        msg = self.br.cv2_to_compressed_imgmsg(msg)
        msg.header = image.header
            
        if queueNum == 1:
            self.image1_msgs.append(msg)
        elif queueNum == 2:
            self.image2_msgs.append(msg)
        elif queueNum == 3:
            self.image3_msgs.append(msg)
        self.sync_messages()

    def imu_callback(self, msg):
        self.imu_msgs.append(msg)
        self.sync_messages()

    def odom_callback(self, msg):
        self.odom_msgs.append(msg)
        self.sync_messages()

    def sync_messages(self):
        while self.image1_msgs and self.image2_msgs and self.image3_msgs and self.imu_msgs and self.odom_msgs:
            image1_time = self.image1_msgs[0].header.stamp.sec + self.image1_msgs[0].header.stamp.nanosec * 1e-9
            image2_time = self.image2_msgs[0].header.stamp.sec + self.image2_msgs[0].header.stamp.nanosec * 1e-9
            image3_time = self.image3_msgs[0].header.stamp.sec + self.image3_msgs[0].header.stamp.nanosec * 1e-9
            imu_time = self.imu_msgs[0].header.stamp.sec + self.imu_msgs[0].header.stamp.nanosec * 1e-9
            odom_time = self.odom_msgs[0].header.stamp.sec + self.odom_msgs[0].header.stamp.nanosec * 1e-9

            # Find the average timestamp
            avg_time = (image1_time + image2_time + image3_time + imu_time + odom_time) / 5.0

            # Calculate time differences
            time_diff_image1 = abs(image1_time - avg_time)
            time_diff_image2 = abs(image2_time - avg_time)
            time_diff_image3 = abs(image3_time - avg_time)
            time_diff_imu = abs(imu_time - avg_time)
            time_diff_odom = abs(odom_time - avg_time)

            # Synchronize if all time differences are within the threshold
            threshhold = 0.05
            if time_diff_image1 < threshhold and time_diff_image2 < threshhold and time_diff_image3 < threshhold and time_diff_imu < threshhold and time_diff_odom < threshhold:
                img1_msg = self.image1_msgs.popleft()
                img2_msg = self.image2_msgs.popleft()
                img3_msg = self.image3_msgs.popleft()
                imu_msg = self.imu_msgs.popleft()
                odom_msg = self.odom_msgs.popleft()

                # Decode the images
                img1_data = np.frombuffer(img1_msg.data, np.uint8)
                img2_data = np.frombuffer(img2_msg.data, np.uint8)
                img3_data = np.frombuffer(img3_msg.data, np.uint8)
                img1 = cv2.imdecode(img1_data, cv2.IMREAD_COLOR)
                img2 = cv2.imdecode(img2_data, cv2.IMREAD_COLOR)
                img3 = cv2.imdecode(img3_data, cv2.IMREAD_COLOR)

                # Stitch the images together using panoramic view
                #stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
                #status, stitched_img = stitcher.stitch([img1, img2, img3])

                #if status != cv2.Stitcher_OK:
                #    print("Error during stitching")
                #    continue

                #stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
                #status, stitched_img = stitcher.stitch([img1, img2])
                #if status != cv2.Stitcher_OK:
                #    print("Error during stitching")
                #    continue
                
                # stitched_img = np.hstack((img1, img2, img3))
                # stitched_img = ImageStitcher.stitch_images([img1, img2, img3])
                
                # Encode the stitched image back to compressed format
                _, stitched_img_encoded = cv2.imencode('.jpg', img2)
                stitched_img_data = stitched_img_encoded.tobytes()                
                
                # img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                img_msg_fields = {"timestamp": image1_time, "data": img2_data}
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
                # Discard the message with the earliest timestamp to find a better match
                min_time = min(image1_time, image2_time, image3_time, imu_time, odom_time)
                if min_time == image1_time:
                    self.image1_msgs.popleft()
                elif min_time == image2_time:
                    self.image2_msgs.popleft()
                elif min_time == image3_time:
                    self.image3_msgs.popleft()
                elif min_time == imu_time:
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
                
                if topic == self.img_topic1:
                    msg = deserialize_message(msg, Image)
                    self.image_callback(msg, 1)
                elif topic == self.img_topic2:
                    msg = deserialize_message(msg, Image)
                    self.image_callback(msg, 2)
                elif topic == self.img_topic3:
                    msg = deserialize_message(msg, Image)
                    self.image_callback(msg, 3)
                elif topic == self.odometry_topic:
                    # Process the odometry messages
                    msg = deserialize_message(msg, Odometry)
                    self.odom_callback(msg)
                elif topic == self.imu_topic:
                    # Process the IMU messages
                    msg = deserialize_message(msg, Imu)
                    self.imu_callback(msg)

                # match topic_type:
                #     case "sensor_msgs/msg/Image":
                #         msg = deserialize_message(msg, Image)
                #         self.image_callback(msg)
                #     case "sensor_msgs/msg/CompressedImage":
                #         msg = deserialize_message(msg, CompressedImage)
                #         self.image_callback(msg)
                #     case "sensor_msgs/msg/CameraInfo":
                #         msg = deserialize_message(msg, CameraInfo)
                #         self.camera_info = msg
                #     case "nav_msgs/msg/Odometry":
                #         if topic == self.odometry_topic:
                #             msg = deserialize_message(msg, Odometry)
                #             self.odom_callback(msg)
                #     case "sensor_msgs/msg/Imu":
                #         if topic == self.imu_topic:
                #             msg = deserialize_message(msg, Imu)
                #             self.imu_callback(msg)

                pbar.update(1)

    def save_data(self):
        if self.VISUAL:
            for i in tqdm(range(len(self.synced_msgs["image"])), desc="Writing video"):
                img_data = self.synced_msgs["image"][i]["data"]
                img = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                height, width, _ = img.shape
                img_save_path = os.path.join(self.BAG_PATH, f"image_{i}.jpg")
                cv2.imwrite(img_save_path, img)
                print(height, width)
                
            # Initialize the video writer
            img_data = self.synced_msgs["image"][0]["data"]
            img = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            height, width, _ = img.shape
            
            frame_size = (width, height)
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_save_path = os.path.join(self.BAG_PATH, "original.mp4")
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, frame_size)

            for i in tqdm(range(len(self.synced_msgs["image"])), desc="Writing video"):
                img_data = self.synced_msgs["image"][i]["data"]
                img = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                video_writer.write(img)

            video_writer.release()
            cprint(f"Video saved successfully: {video_save_path}", "green")

        # Save the data as a pickle file
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        file_path = os.path.join(self.SAVE_PATH, self.BAG_PATH.split("/")[-1] + "_synced.pkl")
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

    # parser.add_argument("--simulation", "-sim", action="store_true", default=False, help="Rosbag is from a Gazebo simulation.")
    args = parser.parse_args()

    cprint(
        f"Bag Path: {args.bag_path}\n" f"Save Path: {args.save_path}\n" f"Visualization: {args.visual}",
        "blue",
    )

    # Process the rosbag
    processor = SynchronizeRosbag(
        bag_path=os.path.normpath(args.bag_path),
        visual=args.visual,
        # simulation=args.simulation
    )
    processor.read_rosbag()
    processor.save_data()
