import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, Imu
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
import argparse
from std_msgs.msg import Header
from homography_from_chessboard import HomographyFromChessboardImage
from camera_intrinsics import CameraIntrinsics
from utils import draw_points, compute_model_chessboard_2d
from bev_costmap import BEVCostmap


class SterlingLocalCostmap(Node):
    def __init__(self, camera_topic, odometry_topic, local_costmap_updates_topic):
        super().__init__("sterling_local_costmap")
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

    def camera_callback(self, msg):
        # When receiving a camera image, update the costmap
        if self.odometry_pose:
            image_data = msg.data
            # TODO: Make a call to the BEVCostmap class to get the terrain costmap
            terrain_costmap = self.get_terrain_preferred_costmap(image_data)
            self.update_costmap(terrain_costmap)

    def odometry_callback(self, msg):
        # Store the odometry message pose
        self.odometry_pose = msg.pose.pose

    def update_costmap(self, terrain_costmap):
        """
        Args:
            terrain_costmap: 2D integer array of terrain cost values (0, 255)
        """
        # Create an OccupancyGrid message
        msg = OccupancyGrid()

        # Fill in the header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()  # Current time
        msg.header.frame_id = "map"  # Frame ID (e.g., 'map' or 'odom')

        # Set the update region
        # TODO: Offset to place region in front of the robot where camera is looking
        msg.x = int(self.odometry_pose.position.x)  # X coordinate of the update region
        msg.y = int(self.odometry_pose.position.y)  # Y coordinate of the update region
        msg.width = len(terrain_costmap[0])  # Width of the update region
        msg.height = len(terrain_costmap)  # Height of the update region

        msg.data = terrain_costmap

        # Publish the update
        self.local_costmap_updates_publisher.publish(msg)
        self.get_logger().info("Published costmap update")


# cb_tile_width = hom_from_cb.cb_tile_width # Chessboard tile width in pixels
# Global costmap resolution = 0.04 m/cell
def get_BEV_image(image, hom_from_cb, patch_size=(128, 128), grid_size=(7, 12), visualize=False):
    # TODO: Use a matrix of corners and apply a *rotation* to align the corners
    # to be perfectly horizontal and apply *translation* to have resulting plane
    # on the grid to be centered and aligned with the bottom image edge
    model_cb = compute_model_chessboard_2d(grid_size[0], grid_size[1], patch_size[0], True)

    annotated_image = image.copy()
    rows, cols = grid_size

    # Manual adjustment that should be automated with the TODO above
    origin_shift = (patch_size[0], patch_size[1] * 2 + 60)

    row_images = []
    for i in range(-rows // 2, rows // 2):
        col_patches = []
        for j in range(-cols // 2, cols // 2):
            x_shift = j * patch_size[0] + origin_shift[0]
            y_shift = i * patch_size[1] + origin_shift[1]

            # Compute the translated homography
            T_shift = np.array([[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]])
            H_shifted = T_shift @ np.linalg.inv(hom_from_cb.H)

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
        cv2.imshow("Stitched BEV Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(args=None):
    camera_topic = "/oakd/oak_d_node/rgb/image_rect_color"
    odometry_topic = "/odometry/filtered"
    local_costmap_updates_topic = "/local_costmap/costmap_updates"

    cb_calibration_image = cv2.imread(os.path.join("homography", "raw_image.jpg"))
    chessboard_homography = HomographyFromChessboardImage(cb_calibration_image, 8, 6)
    H = np.linalg.inv(chessboard_homography.H)
    K, _ = CameraIntrinsics().get_camera_calibration_matrix()

    terrain_encoder_model_path = "../bags/ahg_courtyard_1/models/vis_rep.pt"
    kmeans_model_path = "../bags/ahg_courtyard_1/models/vis_rep.pt"
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

    bev_image = get_BEV_image(cb_calibration_image, chessboard_homography, visualize=False)
    result = BEV_costmap.BEV_to_costmap(bev_image, 128)

    cv2.imshow("Stitched BEV Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # rclpy.init(args=args)
    # costmap_updater = SterlingLocalCostmap(camera_topic, odometry_topic, local_costmap_updates_topic)
    # rclpy.spin(costmap_updater)
    # costmap_updater.destroy_node()
    # rclpy.shutdown()


if __name__ == "__main__":
    main()
