#!/usr/bin/env python3
"""
process_rosbag.py

ROS node that processes a rosbag converting camera, IMU, and odometry data
into a Python dictionary and saves it as a pickle file.

The main functionalities include:
1. Read the rosbag sequentially.
2. Synchronizing the received data.
3. Processing the data to extract patches from images based on odometry information.
4. Saving the processed data into a pickle file.
"""

import os
import pickle

import cv2
import numpy as np
import rclpy
import rosbag2_py
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu
from tqdm import tqdm

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))

# Load the configuration from the YAML file
with open(os.path.join(package_share_directory, "config", "rosbag.yaml"), "r") as file:
    config = yaml.safe_load(file)
    RECORDED_TOPICS = config["recorded_topics"]
    CAMERA_INTRINSICS = config["camera_intrinsics"]
    CAMERA_IMU_TRANSFORM = config["camera_imu_transform"]
    PATCH_PARAMETERS = config["patch_parameters"]

PATCH_SIZE = PATCH_PARAMETERS["patch_size"]
PATCH_EPSILON = 0.5 * PATCH_SIZE * PATCH_SIZE


class ProcessRosbag(Node):
    """
    A class for a ROS node that listens to camera, IMU and legoloam localization info
    and saves the processed data into a pickle file after the rosbag play is finished.
    """

    def __init__(self):
        super().__init__("process_rosbag")

        # Declare ROS parameters with default values
        self.declare_parameter("save_path", os.path.join(ros_ws_dir, "datasets"))
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

        self.camera_info = None

    def read_rosbag(self):
        """
        Reads and processes messages from a ROS2 bag file.
        """
        # Check if the bag file exists
        if not os.path.exists(self.bag_path):
            raise FileNotFoundError(f"Path does not exist: bag_path:={self.bag_path}")

        # Validate the path is a rosbag by checking for metadata.yaml and .db3 file
        yaml_files = [file for file in os.listdir(self.bag_path) if file.endswith(".yaml")]
        db3_files = [file for file in os.listdir(self.bag_path) if file.endswith(".db3")]
        if len(yaml_files) != 1 or len(db3_files) != 1:
            raise FileNotFoundError(
                f"Invalid bag. Bag must contain exactly one .yaml and one .db3 file: {self.bag_path}"
            )

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

                match topic_type:
                    case "sensor_msgs/msg/CompressedImage":
                        msg = deserialize_message(msg, CompressedImage)
                        last_image_msg = msg
                    case "sensor_msgs/msg/CameraInfo":
                        msg = deserialize_message(msg, CameraInfo)
                        self.camera_info = msg
                    case "nav_msgs/msg/Odometry":
                        msg = deserialize_message(msg, Odometry)
                        last_odom_msg = msg
                    case "sensor_msgs/msg/Imu":
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

    def imu_callback(self, msg) -> None:
        """
        Callback function to process IMU messages.
        Args:
            msg (Imu): The IMU message.
        """
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

        # TODO: Check if the orientation from topic is correct (returned w as 0)
        self.imu_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, 1])
        # self.imu_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def image_odom_callback(self, image_msg, odom_msg) -> None:
        """
        Callback function to process synchronized image and odometry messages.
        Args:
            image_msg (CompressedImage): The synchronized image message.
            odom_msg (Odometry): The synchronized odometry message.
        """
        # Extract the yaw angle (rotation around the Z-axis) from the odometry message
        orientation = R.from_quat(
            [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ]
        ).as_euler("XYZ")[-1]

        odom_val = np.asarray([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, orientation])

        self.msg_data["image_msg"].append(image_msg)
        self.msg_data["imu_history"].append(self.imu_buffer.flatten())
        self.msg_data["imu_orientation"].append(self.imu_orientation)
        self.msg_data["odom"].append(odom_val.copy())

    def save_data(self):
        """
        Processes the data stored in 'self.msg_data' and saves it into a pickle file:
        """
        # Dictionary to hold all the processed data
        data = {"patches": [], "imu": []}

        # Buffer to hold the recent 20 BEV images for patch extraction
        buffer = {"bev_img": [], "odom": []}

        C_i, C_i_inv = self.get_camera_intrinsics()

        for i in tqdm(range(len(self.msg_data["image_msg"])), desc="Extracting patches"):
            bev_img, _ = self.camera_imu_homography(
                self.msg_data["imu_orientation"][i], self.msg_data["image_msg"][i], C_i, C_i_inv
            )
            buffer["bev_img"].append(bev_img)
            buffer["odom"].append(self.msg_data["odom"][i])

            curr_odom = self.msg_data["odom"][i]
            patch_list = []

            for j in range(0, len(buffer["bev_img"])):
                prev_image = buffer["bev_img"][j]
                prev_odom = buffer["odom"][j]

                patch, patch_img = self.get_patch_from_odom_delta(curr_odom, prev_odom, prev_image)
                print(f"Patch {patch} extracted from image {j}")

                if patch is not None:
                    patch_list.append(patch)

                    if self.visual:
                        # bev_img = cv2.resize(bev_img, (bev_img.shape[1] // 3, bev_img.shape[0] // 3))
                        # patch_img = cv2.resize(patch_img, (patch_img.shape[1] // 3, patch_img.shape[0] // 3))
                        cv2.imshow(
                            "Current Image <-> Patch Image",
                            np.hstack((bev_img, patch_img)),
                        )
                        cv2.waitKey(5)

                # Collect max of 10 patches
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

    def get_camera_intrinsics(self):
        """
        Get camera intrinsics and its inverse.
        Returns:
            C_i: Camera intrinsic matrix.
            C_i_inv: Inverse of the camera intrinsic matrix.
        """
        if self.camera_info is not None:
            # Extract from topic if available
            fx = self.camera_info.K[0]
            fy = self.camera_info.K[4]
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]
        else:
            # Default use config values
            self.get_logger().warn("\033[0;33mCamera intrinsics not received. Using default values from config.\033[0m")
            fx = CAMERA_INTRINSICS["fx"]
            fy = CAMERA_INTRINSICS["fy"]
            cx = CAMERA_INTRINSICS["cx"]
            cy = CAMERA_INTRINSICS["cy"]

        C_i = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape((3, 3))
        C_i_inv = np.linalg.inv(C_i)
        return C_i, C_i_inv

    def camera_imu_homography(self, orientation_quat, image, C_i, C_i_inv):
        """
        Compute a homography matrix based on camera displacement and orientation changes.
        Args:
            orientation_quat: Orientation quaternion of the IMU.
            image: Compressed image message.
            C_i: Camera intrinsic matrix.
            C_i_inv: Inverse of the camera intrinsic matrix.
        Returns:
            output: Bird's-eye view (BEV) image.
            img: Original image.
        """
        # Extract parameters from the configuration file
        R_cam_imu_angles = CAMERA_IMU_TRANSFORM["R_cam_imu"]
        R_pitch_angles = CAMERA_IMU_TRANSFORM["R_pitch"]
        t1_values = CAMERA_IMU_TRANSFORM["t1"]
        R2_angles = CAMERA_IMU_TRANSFORM["R2"]
        t2_values = CAMERA_IMU_TRANSFORM["t2"]

        # Convert IMU orientation quaternion to Euler angles
        R_imu_world = R.from_quat(orientation_quat)
        R_imu_world = R_imu_world.as_euler("XYZ", degrees=True)

        # Rotation matrix from IMU to camera frame
        R_cam_imu = R.from_euler("xyz", R_cam_imu_angles, degrees=True)
        R_pitch = R.from_euler("XYZ", R_pitch_angles, degrees=True)
        R1 = R_pitch * R_cam_imu
        R1 = R1.as_matrix()
        # Translation vector from IMU to camera frame
        t1 = R1 @ np.array(t1_values).reshape((3, 1))

        # Rotation matrix from camera frame to BEV camera frame
        R2 = R.from_euler("XYZ", R2_angles, degrees=True).as_matrix()
        # Translation vector from camera frame to BEV camera frame
        t2 = R2 @ np.array(t2_values).reshape((3, 1))

        # Normal vector of camera
        n = np.array([0, 0, 1]).reshape((3, 1))
        n1 = R1 @ n

        # Compute the homography matrix
        H12 = ProcessRosbag.homography_camera_displacement(R1, R2, t1, t2, n1)
        homography_matrix = C_i @ H12 @ C_i_inv
        homography_matrix /= homography_matrix[2, 2]

        # Convert the compressed image message to an OpenCV image
        img = np.frombuffer(image.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Homography transformation to create BEV image
        output = cv2.warpPerspective(img, homography_matrix, (img.shape[1], img.shape[0]))
        output = cv2.flip(output, 1)

        return output, img.copy()

    @staticmethod
    def homography_camera_displacement(R1, R2, t1, t2, n1):
        """
        Compute the homography matrix based on camera displacement.
        Args:
            R1: Rotation matrix from IMU to camera frame.
            R2: Rotation matrix from camera frame to BEV camera frame.
            t1: Translation vector from IMU to camera frame.
            t2: Translation vector from camera frame to BEV camera frame.
            n1: Normal vector of camera.
        Returns:
            H12: Homography matrix.
        """
        R12 = R2 @ R1.T
        t12 = R2 @ (-R1.T @ t1) + t2

        d = np.linalg.norm(n1.dot(t1.T))

        H12 = R12 + ((t12 @ n1.T) / d)
        H12 /= H12[2, 2]
        return H12

    def get_patch_from_odom_delta(self, curr_pos, prev_pos, prev_image):
        """
        Extracts a specific patch (sub-image) from a previous image
        based on the current and previous positions and orientations.
        Args:
            curr_pos: Current position (x, y, theta) in the world frame.
            prev_pos: Previous position (x, y, theta) in the world frame.
            prev_image: Previous image from which the patch is to be extracted.
        Returns:
            patch: 3D array with shape (64, 64, 3), representing 64x64 pixel image with 3 color channels (RGB).
            patch_img: Image with the patch corners drawn on it.
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
        if self.visual:
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

        # If the patch has too many zero pixels, return None
        if np.sum(zero_count) > PATCH_EPSILON:
            return None, patch_img

        return patch, patch_img


def main(args=None):
    rclpy.init(args=args)
    node = ProcessRosbag()

    try:
        node.read_rosbag()
        node.save_data()
    except Exception as e:
        node.get_logger().error(f"{e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
