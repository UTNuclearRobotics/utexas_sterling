import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, Imu
from map_msgs.msg import OccupancyGrid, OccupancyGridUpdate
from nav_msgs.msg import Odometry
import argparse


class CostmapUpdater(Node):
    def __init__(self, camera_topic, odometry_topic, local_costmap_updates_topic):
        super().__init__("costmap_updater")
        self.camera_topic = camera_topic
        self.odometry_topic = odometry_topic
        self.local_costmap_updates_topic = local_costmap_updates_topic

        self.local_costmap_updates_publisher = self.create_publisher(OccupancyGridUpdate, self.local_costmap_updates_topic, 10)
        self.camera_subscriber = self.create_subscription(Image, self.camera_topic, self.camera_callback, 10)
        self.odometry_subscriber = self.create_subscription(Odometry, self.odometry_topic, self.odometry_callback, 10)

    def camera_callback(self, msg):
        # When receiving a camera image, update the costmap
        if (self.odometry_pose):
            image_data = msg.data
            terrain_costmap = self.get_terrain_preferred_costmap(image_data)
            self.update_costmap(terrain_costmap)
            
    def odometry_callback(self, msg):
        # Store the odometry message pose
        self.odometry_pose = msg.pose.pose

    def update_costmap(self, terrain_costmap):
        # Create an OccupancyGridUpdate message
        update_msg = OccupancyGridUpdate()

        # Set the update region
        update_msg.x = int(self.odometry_pose.position.x)  # X coordinate of the update region
        update_msg.y = int(self.odometry_pose.position.y)  # Y coordinate of the update region
        update_msg.width = len(terrain_costmap[0])  # Width of the update region
        update_msg.height = len(terrain_costmap)  # Height of the update region

        # Set the updated data (e.g., set all cells to a cost of 100)
        update_msg.data = terrain_costmap

        # Publish the update
        self.local_costmap_updates_publisher.publish(update_msg)
        self.get_logger().info("Published costmap update")


def main(args=None):
    camera_topic = "/oakd/oak_d_node/rgb/image_rect_color"
    odometry_topic = "/odometry/filtered"
    local_costmap_updates_topic = "/local_costmap/costmap_updates"

    rclpy.init(args=args)
    costmap_updater = CostmapUpdater(camera_topic, odometry_topic, local_costmap_updates_topic)
    rclpy.spin(costmap_updater)
    costmap_updater.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
