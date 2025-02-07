import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm
from bev_costmap import BEVCostmap
from homography_from_chessboard import HomographyFromChessboardImage
from robot_data_at_timestep import RobotDataAtTimestep  
import numpy as np
import cv2
from image_stitcher import MapViewer
from collections import defaultdict


class GlobalCostmap:
    def __init__(self, tile_size=1280, cell_size=64, meters_per_pixel=0.0038):
        """
        Initializes a tiled global costmap.

        Args:
            tile_size: The size of each tile (square) in pixels.
            cell_size: Size of each cell in the costmap.
            meters_per_pixel: Conversion factor from real-world meters to pixels.
        """
        self.tile_size = tile_size
        self.cell_size = cell_size
        self.meters_per_pixel = meters_per_pixel

        self.tiles = {}  # Store tiles as {(tile_x, tile_y): np.array}
        self.tile_histograms = {}
        self.global_origin_x = 0
        self.global_origin_y = 0

    def get_tile(self, x, y):
        """
        Retrieves the tile containing (x, y). If the tile does not exist, create it.
        """
        tile_x = x // self.tile_size
        tile_y = y // self.tile_size

        if (tile_x, tile_y) not in self.tiles:
            self.tiles[(tile_x, tile_y)] = np.full((self.tile_size, self.tile_size), 255, dtype=np.uint8)

        return self.tiles[(tile_x, tile_y)], tile_x, tile_y

    def update_cell(self, cell, global_position):
        """
        Updates a 64x64 cell in the appropriate tile of the global costmap based on the most frequently occurring value,
        ensuring that existing values are not overwritten by 255.
        """
        y_offset, x_offset = global_position

        # Compute tile indices
        tile_x = x_offset // self.tile_size
        tile_y = y_offset // self.tile_size

        # Retrieve or create the required tile
        if (tile_x, tile_y) not in self.tiles:
            self.tiles[(tile_x, tile_y)] = np.full((self.tile_size, self.tile_size), 255, dtype=np.uint8)
            self.tile_histograms[(tile_x, tile_y)] = defaultdict(lambda: np.zeros((self.tile_size, self.tile_size), dtype=int))

        tile = self.tiles[(tile_x, tile_y)]
        histogram = self.tile_histograms[(tile_x, tile_y)]

        # Compute position within the tile
        local_x = x_offset % self.tile_size
        local_y = y_offset % self.tile_size

        # Ensure updates fit within tile bounds
        x_end = min(local_x + self.cell_size, self.tile_size)
        y_end = min(local_y + self.cell_size, self.tile_size)

        # Compute valid update region
        tile_roi = tile[local_y:y_end, local_x:x_end]
        new_cell_roi = cell[:(y_end - local_y), :(x_end - local_x)]

        # Update frequency histogram (EXCLUDE 255)
        unique_values = np.unique(new_cell_roi)
        unique_values = unique_values[unique_values != 255]  # Ignore 255 in histogram updates

        for value in unique_values:
            mask = (new_cell_roi == value)
            histogram[value][local_y:y_end, local_x:x_end] += mask.astype(int)

        # Assign each cell its most frequent value
        max_counts = np.zeros_like(tile_roi)
        most_frequent_values = tile_roi.copy()  # Start with existing values instead of filling with 255

        for value, count_map in histogram.items():
            mask = count_map[local_y:y_end, local_x:x_end] > max_counts
            most_frequent_values[mask] = value
            max_counts[mask] = count_map[local_y:y_end, local_x:x_end][mask]

        # Apply most frequent values to the tile (ENSURE 255 DOES NOT OVERWRITE EXISTING DATA)
        mask_255 = most_frequent_values != 255  # Only update where new value is not 255
        tile[local_y:y_end, local_x:x_end][mask_255] = most_frequent_values[mask_255]

    def process_odometry_and_update(self, local_costmap, odom_matrix):
        """
        Processes odometry data, places local costmap into the correct tile, and updates accordingly.
        """
        # Extract translation from odometry
        t_x, t_y = -odom_matrix[1, 3], odom_matrix[0, 3]

        # Convert meters to pixels
        x_pixel = int(t_x / self.meters_per_pixel) + self.global_origin_x
        y_pixel = int(t_y / self.meters_per_pixel) + self.global_origin_y

        # Extract yaw
        cos_theta = odom_matrix[0, 0]
        sin_theta = odom_matrix[1, 0]
        theta_degrees = np.degrees(np.arctan2(sin_theta, cos_theta))

        # Convert local costmap to grayscale if needed
        if len(local_costmap.shape) == 3:
            local_costmap = cv2.cvtColor(local_costmap, cv2.COLOR_BGR2GRAY)

        # Get costmap dimensions
        h, w = local_costmap.shape
        num_cells_x = w // self.cell_size
        num_cells_y = h // self.cell_size

        # Rotation matrix
        rotation_matrix = np.array([
            [cos_theta, sin_theta],  
            [-sin_theta, cos_theta]
        ])

        # Process and update costmap in tiles
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                cell = local_costmap[i * self.cell_size: (i + 1) * self.cell_size,
                                     j * self.cell_size: (j + 1) * self.cell_size]

                local_pos = np.array([
                    (j - num_cells_x // 2) * self.cell_size,  
                    (i - num_cells_y // 2) * self.cell_size
                ])
                rotated_pos = rotation_matrix @ local_pos
                cell_x = x_pixel + int(rotated_pos[0])
                cell_y = y_pixel - int(rotated_pos[1])

                cell_x = (cell_x // self.cell_size) * self.cell_size
                cell_y = (cell_y // self.cell_size) * self.cell_size

                rotated_cell = self.rotate_cell_bottom_center(cell, theta_degrees)

                # Update the correct tile
                self.update_cell(rotated_cell, (cell_x, cell_y))

    def rotate_cell_bottom_center(self, cell, theta_degrees):
        """
        Rotates a 64x64 cell around its bottom-center point.
        """
        h, w = cell.shape
        center = (w // 2, h)

        rotation_matrix = cv2.getRotationMatrix2D(center, theta_degrees, 1.0)
        rotated_cell = cv2.warpAffine(cell, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

        rotated_cell[rotated_cell == 0] = np.max(rotated_cell)
        return rotated_cell

    def get_combined_costmap(self):
        """
        Returns a combined view of all active tiles.
        """
        if not self.tiles:
            return None

        min_x = min(tile_x for tile_x, _ in self.tiles) * self.tile_size
        min_y = min(tile_y for _, tile_y in self.tiles) * self.tile_size
        max_x = (max(tile_x for tile_x, _ in self.tiles) + 1) * self.tile_size
        max_y = (max(tile_y for _, tile_y in self.tiles) + 1) * self.tile_size

        # Create combined costmap
        combined_map = np.full((max_y - min_y, max_x - min_x), 255, dtype=np.uint8)

        for (tile_x, tile_y), tile in self.tiles.items():
            x_start = (tile_x * self.tile_size) - min_x
            y_start = (tile_y * self.tile_size) - min_y
            combined_map[y_start:y_start + self.tile_size, x_start:x_start + self.tile_size] = tile

        return combined_map


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Load the image
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    image = cv2.imread(os.path.join(image_dir, image_file))

    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)
    #H = np.linalg.inv(chessboard_homography.H)  # get_homography_image_to_model()
    H, dsize,_ = chessboard_homography.plot_BEV_full(image)
    parser = argparse.ArgumentParser(description="Generate and update a global BEV cost map.")
    parser.add_argument("-b", type=str, required=True, help="Bag directory with synchronized pickle file inside.")
    args = parser.parse_args()

    bag_path = args.b
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    synced_pkl = [file for file in os.listdir(bag_path) if file.endswith("_synced.pkl")]
    if len(synced_pkl) != 1:
        raise FileNotFoundError(f"Synced pickle file not found in: {bag_path}")
    synced_pkl_path = os.path.join(bag_path, synced_pkl[0])

    # Initialize BEV Costmap
    viz_encoder_path = "models/vis_rep.pt"
    kmeans_path = "scripts/clusters/kmeans_model.pkl"
    scaler_path = "scripts/clusters/scaler.pkl"

    preferences = {
        0: 50,   # Aggregate Concrete
        1: 225,  # Metal
        2: 0,    # Smooth Concrete
        3: 225,  # Grass
        4: 100,  # Aggregate with leaves
        5: 225,  # Grass
        6: 0     # Smooth Concrete
    }

    bev_costmap = BEVCostmap(viz_encoder_path, kmeans_path, scaler_path, preferences)
    global_costmap = GlobalCostmap()

    # Process each timestep
    robot_data = RobotDataAtTimestep(synced_pkl_path)
    odom_prev = np.eye(4, dtype=np.float32)  # Initial odometry is identity

    for timestep in tqdm(range(0, robot_data.getNTimesteps()), desc="Processing costmaps"):
        cur_img = robot_data.getImageAtTimestep(timestep)
        odom_cur = robot_data.getOdomAtTimestep(timestep)
        robot_x, robot_y = -odom_cur[1, 3], -odom_cur[0, 3]

        bev_img = cv2.warpPerspective(cur_img, H, dsize)
        costmap = bev_costmap.BEV_to_costmap(bev_img, 64)
        visualize_cost = bev_costmap.visualize_costmap(costmap, 64)

        global_costmap.process_odometry_and_update(visualize_cost, odom_cur)

        #if timestep % 5 == 0:
            #updated_costmap = global_costmap.get_combined_costmap()
            #cv2.namedWindow("Global Cost Map", cv2.WINDOW_NORMAL)
            #cv2.imshow("Global Cost Map", updated_costmap)
            #cv2.waitKey(10)

    final_map = global_costmap.get_combined_costmap()
    viewer = MapViewer(final_map)
    viewer.show_map()
    viewer.save_full_map(filename="full_costmap.png")