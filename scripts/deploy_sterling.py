import argparse
import os

import cv2
import numpy as np
import rclpy
from bev_costmap import BEVCostmap
from camera_intrinsics import CameraIntrinsics
from geometry_msgs.msg import Pose
from homography_from_chessboard import HomographyFromChessboardImage
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, Point, Quaternion
from std_msgs.msg import Header
from utils import compute_model_chessboard_2d, draw_points


class SterlingLocalCostmap(Node):
    def __init__(self, camera_topic, odometry_topic, local_costmap_updates_topic, cb_homography, BEV_costmap):
        super().__init__("sterling_local_costmap")
        # ROS topics
        self.camera_topic = camera_topic
        self.odometry_topic = odometry_topic
        self.local_costmap_updates_topic = local_costmap_updates_topic

        # ROS publishers
        self.local_costmap_updates_publisher = self.create_publisher(
            OccupancyGrid, self.local_costmap_updates_topic, 10
        )

        # ROS subscribers
        self.camera_subscriber = self.create_subscription(Image, self.camera_topic, self.camera_callback, 10)
        self.odometry_subscriber = self.create_subscription(Odometry, self.odometry_topic, self.odometry_callback, 10)

        self.H_inv = np.linalg.inv(cb_homography.H)
        self.get_terrain_preferred_costmap = BEV_costmap
        
        self.odometry_pose = None

    def camera_callback(self, msg):
        # When receiving a camera image, update the costmap
        if self.odometry_pose:
            image_data = msg.data
            bev_image = get_BEV_image(image_data, self.H_inv, (128, 128), (7, 12))
            
            # TODO: Treat unconfident terrain as unknown (-1)
            terrain_costmap = self.get_terrain_preferred_costmap(bev_image, 128)
            terrain_costmap = (terrain_costmap / 2.55).astype(np.int8)  # Max value is 255, so divide by 2.55 to get 100 and convert to int8
            
            # Make it 1D 
            terrain_costmap_flat = [cell for row in terrain_costmap for cell in row]
            self.update_costmap(terrain_costmap_flat)

    def odometry_callback(self, msg):
        # Store the odometry message pose
        self.odometry_pose = msg.pose.pose

    def update_costmap(self, terrain_costmap):
        width = len(terrain_costmap[0])
        height = len(terrain_costmap)
        
        # Create an OccupancyGrid message
        msg = OccupancyGrid()

        # Fill in the header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # Fill in the map metadata
        msg.info = MapMetaData()
        # map_load_time # The time at which the map was loaded
        msg.info.resolution # The map resolution [m/cell]
        msg.info.width = width # Map width [cells]
        msg.info.height = height # Map height [cells]
        
        # Offset of the top left of the costmap relative to base_link
        x_offset = 0.23 * (width / 2)
        y_offset = 0.23 * height
    
        # The origin of the map [m, m, rad]. This is the real-world pose of the cell (0,0) in the map.
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.odometry_pose.position.x + x_offset
        msg.info.origin.position.y = self.odometry_pose.position.y + y_offset
        msg.info.origin.position.z = self.odometry_pose.position.z
        msg.info.origin.orientation = self.odometry_pose.orientation

        # The map data, in row-major order, starting with (0,0).
        # Occupancy probabilities are in the range [0,100]. Unknown is -1.
        msg.data = terrain_costmap

        # Publish the update
        self.local_costmap_updates_publisher.publish(msg)
        # self.get_logger().info("Published costmap update")


def align_corners(image, patch_size, grid_size, H_inv):
    """
    Align the corners to be perfectly horizontal and aligned with the bottom image edge
    """
    # TODO: Use a matrix of corners and apply a *rotation* to align the corners
    # to be perfectly horizontal and apply *translation* to have resulting plane
    # on the grid to be centered and aligned with the bottom image edge
    image_width, image_height = image.shape[:2]
    rows, cols = grid_size
    patch_width, patch_height = patch_size

    model_cb = compute_model_chessboard_2d(grid_size[0], grid_size[1], patch_size[0], False)

    # Homogenous corners of the region of interest
    corners = np.array(
        [
            [0, 0, 1],  # Top-left
            [rows * patch_width, 0, 1],  # Top-right
            [rows * patch_width, cols * patch_height, 1],  # Bottom-right
            [0, cols * patch_height, 1],  # Bottom-left
        ],
        dtype=np.float32,
    ).T  # Shape: (3, 4)

    # Compute the inverse homography to map patch corners back to the original image
    corners_on_image = H_inv @ corners  # Shape: (3, 4)

    # Normalize homogeneous coordinates (x, y, w) -> (x/w, y/w)
    corners_on_image = corners_on_image[:2] / corners_on_image[2:3]  # Shape: (2, 4)
    corners_on_image = corners_on_image.T  # Shape: (4, 2), each row is (x, y)

    # Compute rotation to make bottom row horizontal
    bottom_edge_vec = corners_on_image[-1] - corners_on_image[-2]  # Left to right of bottom row
    angle = np.arctan2(bottom_edge_vec[1], bottom_edge_vec[0])
    rotation_angle = -angle  # Rotate to align with x-axis

    # Rotation matrix
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Apply rotation around centroid
    centroid = np.mean(model_cb, axis=0)
    corners_centered = model_cb - centroid
    corners_rotated = corners_centered @ rotation_matrix.T
    corners_rotated += centroid

    # Compute translation to center and align with bottom edge
    bottom_row_rotated = corners_rotated[(rows - 1) * cols : rows * cols]
    min_x, min_y = np.min(corners_rotated, axis=0)
    max_x, max_y = np.max(corners_rotated, axis=0)
    width = max_x - min_x
    height = max_y - min_y

    # Target position
    target_x = (image_width - width) / 2  # Center horizontally
    bottom_row_y = np.mean(bottom_row_rotated[:, 1])  # Average y of bottom row
    target_y = image_height - (max_y - bottom_row_y)  # Align bottom row

    # Translation vector
    translation = np.array([target_x - min_x, target_y - min_y])

    # Apply translation
    corners_final = corners_rotated + translation

    print(corners_final)
    copy = image.copy()
    for corner in corners_final:
        x, y = int(corner[0]), int(corner[1])
        copy = cv2.circle(copy, (x, y), 5, (0, 0, 255), -1)  # Draw red dots
    cv2.imshow("Dots", copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners_final


def get_BEV_image(image, hom_from_cb, patch_size=(128, 128), grid_size=(7, 12), visualize=False):
    annotated_image = image.copy()
    H_inv = np.linalg.inv(hom_from_cb.H)
    rows, cols = grid_size
    patch_width, patch_height = patch_size

    # TODO: Don't make this adjust manual
    origin_shift = (patch_size[0], patch_size[1] * 2 + 60)

    row_images = []
    for i in range(-rows // 2, rows // 2):
        col_patches = []
        for j in range(-cols // 2, cols // 2):
            x_shift = j * patch_size[0] + origin_shift[0]
            y_shift = i * patch_size[1] + origin_shift[1]

            # Compute the translated homography
            T_shift = np.array([[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]])
            H_shifted = T_shift @ H_inv

            # Warp and resize the patch
            cur_patch = cv2.warpPerspective(image, H_shifted, dsize=patch_size)
            if cur_patch.shape != patch_size:
                cur_patch = cv2.resize(cur_patch, patch_size)
            col_patches.append(cur_patch)

            if visualize:
                annotated_image = draw_points(annotated_image, H_shifted, patch_size, color=(0, 255, 0), thickness=2)

        row_image = cv2.hconcat(col_patches[::-1])
        row_images.append(row_image)
    stitched_image = cv2.vconcat(row_images[::-1])
    print(f"Stitched image size: {stitched_image.shape}")

    if visualize:
        cv2.imshow("Current Image with patches", annotated_image)

        # Draw the green grid lines
        for i in range(rows + 1):
            start_point = (0, i * patch_height)
            end_point = (cols * patch_width, i * patch_height)
            stitched_image = cv2.line(stitched_image, start_point, end_point, (0, 255, 0), 2)
        for j in range(cols + 1):
            start_point = (j * patch_width, 0)
            end_point = (j * patch_width, rows * patch_height)
            stitched_image = cv2.line(stitched_image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Stitched BEV Image", stitched_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return stitched_image


def main(args=None):
    camera_topic = "/oakd/oak_d_node/rgb/image_rect_color"
    odometry_topic = "/odometry/filtered"
    local_costmap_updates_topic = "/local_costmap/costmap_updates"

    cb_calibration_image = cv2.imread(os.path.join("homography", "raw_image.jpg"))
    chessboard_homography = HomographyFromChessboardImage(cb_calibration_image, 8, 6)
    H = np.linalg.inv(chessboard_homography.H)
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    terrain_encoder_model_path = "../bags/2_20_ahg_courtyard_1_testing_terrain_rep.pt"
    kmeans_model_path = "../bags/sim_kmeans_model.pkl"
    preferences = {
        # Black: 0, White: 255
        0: 50,  # Cluster 0: Agg, leaves
        1: 0,  # Cluster 1: Smooth concrete
        2: 100,  # Cluster 2: Smooth concrete
        3: 0,  # Cluster 3: Agg
        4: 225,  # Cluster 4: Aggregate concrete, leaves
        5: 50,  # Cluster 5: Grass
        6: 0,  # Cluster 6: Smooth concrete
    }
    BEV_costmap = BEVCostmap(terrain_encoder_model_path, kmeans_model_path, preferences)

    # bev_image = get_BEV_image(cb_calibration_image, chessboard_homography, (128, 128), (7, 12), visualize=True)
    
    # result = BEV_costmap.BEV_to_costmap(bev_image, 128)

    # cv2.imshow("Stitched BEV Image", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rclpy.init(args=args)
    costmap_updater = SterlingLocalCostmap(camera_topic, odometry_topic, local_costmap_updates_topic, chessboard_homography, BEV_costmap)
    rclpy.spin(costmap_updater)
    costmap_updater.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
