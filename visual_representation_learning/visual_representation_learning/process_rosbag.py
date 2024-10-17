#!/usr/bin/env python3
"""
process_rosbag.py

ROS node that listens to camera, IMU, and odometry information, processes the data,
and saves it into a pickle file.

The main functionalities include:
1. Subscribing to ROS topics to receive IMU, camera, and odometry data.
2. Synchronizing the received data.
3. Processing the data to extract patches from images based on odometry information.
4. Saving the processed data into a pickle file.
"""

import os
import pickle
import pdb
import cv2
import numpy as np
import rclpy
import rosbag2_py
import tf2_ros
import yaml
from rclpy.serialization import deserialize_message

from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Imu
from tqdm import tqdm
from termcolor import cprint

package_share_directory = get_package_share_directory("visual_representation_learning")

# Load the configuration from the YAML file
with open(os.path.join(package_share_directory, "config", "rosbag.yaml"), "r") as file:
    config = yaml.safe_load(file)
    RECORDED_TOPICS = config["recorded_topics"]
    CAMERA_INTRINSICS = config["camera_intrinsics"]
    PATCH_PARAMETERS = config["patch_parameters"]


# Function to get camera intrinsics and its inverse
def get_camera_intrinsics():
    fx = CAMERA_INTRINSICS["fx"]
    fy = CAMERA_INTRINSICS["fy"]
    cx = CAMERA_INTRINSICS["cx"]
    cy = CAMERA_INTRINSICS["cy"]

    C_i = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape((3, 3))
    C_i_inv = np.linalg.inv(C_i)
    return C_i, C_i_inv


C_i, C_i_inv = get_camera_intrinsics()


PATCH_SIZE = PATCH_PARAMETERS["patch_size"]
PATCH_EPSILON = 0.5 * PATCH_SIZE * PATCH_SIZE
ACTUATION_LATENCY = PATCH_PARAMETERS["actuation_latency"]


class ProcessRosbag(Node):
    """
    A class for a ROS node that listens to camera, IMU and legoloam localization info
    and saves the processed data into a pickle file after the rosbag play is finished.
    """

    def __init__(self):
        super().__init__("process_rosbag")

        # Declare ROS parameters with default values
        self.declare_parameter("save_path", os.path.expanduser("~/utexas_sterling_ws/pickles"))
        self.declare_parameter("bag_path", "")
        self.declare_parameter("visual", False)

        # Retrieve the parameter values
        self.save_path = self.get_parameter("save_path").get_parameter_value().string_value
        self.bag_path = self.get_parameter("bag_path").get_parameter_value().string_value
        self.visual = self.get_parameter("visual").get_parameter_value().bool_value

        self.imu_buffer = np.zeros((200, 6), dtype=np.float32)

        self.msg_data = {
            "image_msg": [],
            "imu_history": [],
            "imu_orientation": [],
            "odom": [],
        }

        # self.tf_buffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def read_rosbag(self):
        """
        Reads and processes messages from a ROS2 bag file.

        The main functionalities include:
        1. Sets up storage and converter options for reading the rosbag.
        2. Opens the rosbag using a SequentialReader.
        3. Iterates through the messages in the rosbag and processes them based on their topic types.
        4. Synchronizes image and odometry messages based on their timestamps.
        """

        # Check if the bag file exists
        if not os.path.exists(self.bag_path):
            raise FileNotFoundError(f"Bag file does not exist: bag_path:={self.bag_path}")

        # Set up storage and converter options for reading the rosbag
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py._storage.ConverterOptions("", "")
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # Print rosbag metadata
        metadata = reader.get_metadata()
        self.get_logger().info(f"{metadata}")

        # Iterator variables
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        last_image_msg = None
        last_odom_msg = None

        # Iterate through the messages in the rosbag
        with tqdm(total=metadata.message_count, desc="Reading rosbag messages") as pbar:
            while reader.has_next():
                (topic, msg, t) = reader.read_next()
                topic_type = type_map.get(topic)

                if topic_type == "sensor_msgs/msg/CompressedImage":
                    msg = deserialize_message(msg, CompressedImage)
                    last_image_msg = msg
                elif topic_type == "nav_msgs/msg/Odometry":
                    msg = deserialize_message(msg, Odometry)
                    last_odom_msg = msg
                elif topic_type == "sensor_msgs/msg/Imu":
                    msg = deserialize_message(msg, Imu)
                    self.imu_callback(msg)

                # Synchronize messages based on timestamps
                if last_image_msg and last_odom_msg:
                    # Calculate time difference
                    image_time = last_image_msg.header.stamp.sec + last_image_msg.header.stamp.nanosec * 1e-9
                    odom_time = last_odom_msg.header.stamp.sec + last_odom_msg.header.stamp.nanosec * 1e-9
                    time_diff = abs(image_time - odom_time)

                    if time_diff < 0.05:
                        self.image_odom_callback(last_image_msg, last_odom_msg)
                        last_image_msg = None
                        last_odom_msg = None

                pbar.update(1)

    def imu_callback(self, msg):
        self.imu_buffer = np.roll(self.imu_buffer, -1, axis=0)
        self.imu_buffer[-1] = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ]
        )
        
        # TODO: Check if the orientation is correct
        self.imu_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, 1])
        # self.imu_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def image_odom_callback(self, image, odom):
        # Extract the yaw angle (rotation around the Z-axis) from the odometry message
        orientation = R.from_quat(
            [
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ]
        ).as_euler("XYZ")[-1]

        odom_val = np.asarray([odom.pose.pose.position.x, odom.pose.pose.position.y, orientation])

        self.msg_data["image_msg"].append(image)
        self.msg_data["imu_history"].append(self.imu_buffer.flatten())
        self.msg_data["imu_orientation"].append(self.imu_orientation)
        self.msg_data["odom"].append(odom_val.copy())

    def get_localization(self):
        try:
            self.trans = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0),
            )
        except Exception as e:
            self.trans = None
            self.get_logger().error(str(e))

    def display_images(self):
        """
        Debugging function to loops through the 'image_msg' list and displays each image continuously.
        """
        for image_msg in self.msg_data["image_msg"]:
            try:
                # Convert the compressed image message to an OpenCV image
                img = np.frombuffer(image_msg.data, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)

                # Check if the image is correctly converted
                if img is None:
                    self.get_logger().error("Failed to decode compressed image.")
                    continue

                # Display the image
                cv2.imshow("bev_Img", img)
                cv2.waitKey(10)

            except Exception as e:
                self.get_logger().error(f"Failed to convert and display image: {e}")

        # Close all OpenCV windows
        cv2.destroyAllWindows()

    def save_data(self):
        """
        Processes the data stored in 'msg_data' and saves it into a pickle file:

        Images: Processed images or bird's-eye view (BEV) images.
        IMU Data: Flattened IMU data from both Kinect and sensors.
        Odometry Data: Position and orientation data from odometry.
        Orientation Data: Orientation data from the robot.
        """

        # Dictionary to hold all the processed data
        data = {"patches": [], "imu": []}

        # Buffer to hold the recent 20 BEV images for patch extraction
        buffer = {"bev_img": [], "odom": []}

        for i in tqdm(range(len(self.msg_data["image_msg"])), desc="Extracting patches"):
            bev_img, _ = ProcessRosbag.camera_imu_homography(
                self.msg_data["imu_orientation"][i],
                self.msg_data["image_msg"][i],
            )
            buffer["bev_img"].append(bev_img)
            buffer["odom"].append(self.msg_data["odom"][i])

            curr_odom = self.msg_data["odom"][i]
            patch_list = []

            for j in range(0, len(buffer["bev_img"])):
                prev_image = buffer["bev_img"][j]
                prev_odom = buffer["odom"][j]

                patch, patch_img = ProcessRosbag.get_patch_from_odom_delta(
                    curr_odom, prev_odom, prev_image, visualize=self.visual
                )

                if patch is not None:
                    patch_list.append(patch)

                    if self.visual:
                        # bev_img = cv2.resize(bev_img, (bev_img.shape[1] // 3, bev_img.shape[0] // 3))
                        # patch_img = cv2.resize(patch_img, (patch_img.shape[1] // 3, patch_img.shape[0] // 3))
                        cv2.imshow(
                            "BEV Image <-> Patch Image",
                            np.hstack((prev_image, patch_img)),
                        )
                        cv2.waitKey(5)

                if len(patch_list) >= 10:
                    break

            # Remove the oldest image and odometry data from the buffer
            while len(buffer["bev_img"]) > 20:
                buffer["bev_img"].pop(0)
                buffer["odom"].pop(0)

            if len(patch_list) > 0:
                # self.get_logger().info(f"Num patches : {len(patch_list)}")
                data["patches"].append(patch_list)
                data["imu"].append(self.msg_data["imu_history"][i])

        # Ensure the output directory exists
        os.makedirs(self.save_path, exist_ok=True)

        # Construct the full file path
        file_path = os.path.join(self.save_path, self.bag_path.split("/")[-1] + ".pkl")

        try:
            # Open the file in write-binary mode and dump the data
            with open(file_path, "wb") as file:
                pickle.dump(data, file)
            self.get_logger().info(f"Data saved successfully to {file_path}")
        except Exception as e:
            self.get_logger().info(f"Failed to save data to {file_path}: {e}")

    @staticmethod
    def camera_imu_homography(orientation_quat, image):
        R_imu_world = R.from_quat(orientation_quat)
        R_imu_world = R_imu_world.as_euler("XYZ", degrees=True)

        R_cam_imu = R.from_euler("xyz", [-90, 90, 0], degrees=True)
        R_pitch = R.from_euler("XYZ", [26.5, 0, 0], degrees=True)

        R1 = R_pitch * R_cam_imu
        R1 = R1.as_matrix()
        t1 = R1 @ np.array([0.0, 0.0, 0.75]).reshape((3, 1))

        R2 = R.from_euler("XYZ", [-180, 0, 90], degrees=True).as_matrix()
        t2 = R2 @ np.array([4.20, 0.0, 6.0]).reshape((3, 1))

        n = np.array([0, 0, 1]).reshape((3, 1))
        n1 = R1 @ n

        H12 = ProcessRosbag.homography_camera_displacement(R1, R2, t1, t2, n1)
        homography_matrix = C_i @ H12 @ C_i_inv
        homography_matrix /= homography_matrix[2, 2]

        # Convert the compressed image message to an OpenCV image
        img = np.frombuffer(image.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        return img, img.copy()

        # TODO: Fix the homography transformation to create BEV image
        output = cv2.warpPerspective(img, homography_matrix, (img.shape[1], img.shape[0]))
        output = cv2.flip(output, 1)

        return output, img.copy()

    @staticmethod
    def get_patch_from_odom_delta(curr_pos, prev_pos, prev_image, visualize=False):
        """
        Extracts a specific patch (sub-image) from a previous image 
        based on the current and previous positions and orientations.
        """
        
        # Convert current position to homogeneous coordinates
        curr_pos_np = np.array([curr_pos[0], curr_pos[1], 1])
        
        # Get image dimensions
        image_height, image_width = prev_image.shape[:2]

        # Create transformation matrix for previous position
        prev_pos_transform = np.zeros((3, 3))
        prev_pos_transform[:2, :2] = R.from_euler("XYZ", [0, 0, prev_pos[2]]).as_matrix()[:2, :2]
        prev_pos_transform[:, 2] = np.array([prev_pos[0], prev_pos[1], 1]).reshape((3))

        # Invert the transformation matrix
        inv_pos_transform = np.linalg.inv(prev_pos_transform)

        # Compute current Z rotation matrix
        curr_z_rotation = R.from_euler("XYZ", [0, 0, curr_pos[2]]).as_matrix()

        # Define patch corners in the current frame
        patch_corners = [
            curr_pos_np + curr_z_rotation @ np.array([0.3, 0.3, 0]),
            curr_pos_np + curr_z_rotation @ np.array([0.3, -0.3, 0]),
            curr_pos_np + curr_z_rotation @ np.array([-0.3, -0.3, 0]),
            curr_pos_np + curr_z_rotation @ np.array([-0.3, 0.3, 0]),
        ]

        # Transform patch corners to the previous frame
        patch_corners_prev_frame = [
            inv_pos_transform @ patch_corners[0],
            inv_pos_transform @ patch_corners[1],
            inv_pos_transform @ patch_corners[2],
            inv_pos_transform @ patch_corners[3],
        ]
        
        # Scale patch corners
        SCALING_FACTOR = 132.003788 * (image_width / 1024)  # Adjust scaling factor based on image width
        scaled_patch_corners = [
            (patch_corners_prev_frame[0] * SCALING_FACTOR).astype(np.int32),
            (patch_corners_prev_frame[1] * SCALING_FACTOR).astype(np.int32),
            (patch_corners_prev_frame[2] * SCALING_FACTOR).astype(np.int32),
            (patch_corners_prev_frame[3] * SCALING_FACTOR).astype(np.int32),
        ]

        # Define center point for the image frame
        # CENTER = np.array((1024 - 20, (768 - 55) * 2))

        # Transform patch corners to the image frame
        CENTER = np.array((image_width // 2, image_height // 2))
        patch_corners_image_frame = [
            CENTER + np.array((-scaled_patch_corners[0][1], -scaled_patch_corners[0][0])),
            CENTER + np.array((-scaled_patch_corners[1][1], -scaled_patch_corners[1][0])),
            CENTER + np.array((-scaled_patch_corners[2][1], -scaled_patch_corners[2][0])),
            CENTER + np.array((-scaled_patch_corners[3][1], -scaled_patch_corners[3][0])),
        ]

        patch_img = None
        if visualize:
            patch_img = prev_image.copy()

            # Draw lines between patch corners
            cv2.line(
                patch_img,
                (patch_corners_image_frame[0][0], patch_corners_image_frame[0][1]),
                (patch_corners_image_frame[1][0], patch_corners_image_frame[1][1]),
                (0, 255, 0),
                2,
            )
            cv2.line(
                patch_img,
                (patch_corners_image_frame[1][0], patch_corners_image_frame[1][1]),
                (patch_corners_image_frame[2][0], patch_corners_image_frame[2][1]),
                (0, 255, 0),
                2,
            )
            cv2.line(
                patch_img,
                (patch_corners_image_frame[2][0], patch_corners_image_frame[2][1]),
                (patch_corners_image_frame[3][0], patch_corners_image_frame[3][1]),
                (0, 255, 0),
                2,
            )
            cv2.line(
                patch_img,
                (patch_corners_image_frame[3][0], patch_corners_image_frame[3][1]),
                (patch_corners_image_frame[0][0], patch_corners_image_frame[0][1]),
                (0, 255, 0),
                2,
            )

        # Compute perspective transform matrix
        persp = cv2.getPerspectiveTransform(
            np.float32(patch_corners_image_frame),
            np.float32([[0, 0], [63, 0], [63, 63], [0, 63]]),
        )

        # Warp the previous image to extract the patch
        patch = cv2.warpPerspective(prev_image, persp, (64, 64))

        # Check for zero pixels in the extracted patch
        zero_count = np.logical_and(
            np.logical_and(patch[:, :, 0] == 0, patch[:, :, 1] == 0),
            patch[:, :, 2] == 0,
        )

        # TODO: If the patch has too many zero pixels, return None
        # if np.sum(zero_count) > PATCH_EPSILON:
        #     return None, patch_img

        return patch, patch_img

    @staticmethod
    def homography_camera_displacement(R1, R2, t1, t2, n1):
        R12 = R2 @ R1.T
        t12 = R2 @ (-R1.T @ t1) + t2

        d = np.linalg.norm(n1.dot(t1.T))

        H12 = R12 + ((t12 @ n1.T) / d)
        H12 /= H12[2, 2]
        return H12


def main(args=None):
    rclpy.init(args=args)
    node = ProcessRosbag()

    try:
        node.read_rosbag()
        # node.display_images()
        node.save_data()
    except Exception as e:
        node.get_logger().error(f"{e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
