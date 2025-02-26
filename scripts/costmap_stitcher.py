import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm
from bev_costmap import BEVCostmap
from homography_matrix import HomographyMatrix
from robot_data_at_timestep import RobotDataAtTimestep  
from image_stitcher import MapViewer
from collections import defaultdict
from homography_utils import plot_BEV_full


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
        Processes odometry data, places the local costmap into the correct global tile, and updates accordingly.
        """
        # Extract translation (note the sign flip) and convert to pixel coordinates.
        t_x, t_y = -odom_matrix[1, 3], odom_matrix[0, 3]
        x_pixel = int(t_x / self.meters_per_pixel) + self.global_origin_x
        y_pixel = int(t_y / self.meters_per_pixel) + self.global_origin_y

        # Extract yaw (rotation) in degrees.
        cos_theta = odom_matrix[0, 0]
        sin_theta = odom_matrix[1, 0]
        theta_degrees = np.degrees(np.arctan2(sin_theta, cos_theta))

        # Convert to grayscale if needed.
        if local_costmap.ndim == 3:
            local_costmap = cv2.cvtColor(local_costmap, cv2.COLOR_BGR2GRAY)

        h, w = local_costmap.shape
        num_cells_x = w // self.cell_size
        num_cells_y = h // self.cell_size

        # Build the rotation matrix for mapping local costmap positions to global positions.
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta,  cos_theta]])

        # Vectorize computation of local positions.
        # Create a grid of cell indices.
        j_idx, i_idx = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        # Compute local cell center positions relative to the center of the costmap.
        local_pos_x = (j_idx - num_cells_x // 2) * self.cell_size
        local_pos_y = (num_cells_y // 2 - i_idx) * self.cell_size
        local_positions = np.stack((local_pos_x, local_pos_y), axis=-1)  # shape: (num_cells_y, num_cells_x, 2)

        # Rotate all local positions at once.
        rotated_positions = (rotation_matrix @ local_positions.reshape(-1, 2).T).T.reshape(num_cells_y, num_cells_x, 2)
        # Convert to global pixel coordinates (and then align to cell boundaries).
        cell_coords_x = ((x_pixel + rotated_positions[..., 0].astype(np.int32)) // self.cell_size) * self.cell_size
        cell_coords_y = ((y_pixel + rotated_positions[..., 1].astype(np.int32)) // self.cell_size) * self.cell_size

        # Pre-compute the rotation matrix for individual cell rotation.
        cell_center = (self.cell_size // 2, self.cell_size)  # bottom-center point
        rot_mat = cv2.getRotationMatrix2D(cell_center, theta_degrees, 1.0)

        # Process each cell.
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                # Extract the current cell.
                cell = local_costmap[i * self.cell_size: (i + 1) * self.cell_size,
                                    j * self.cell_size: (j + 1) * self.cell_size]
                # Rotate the cell.
                rotated_cell = cv2.warpAffine(cell, rot_mat, (self.cell_size, self.cell_size),
                                                flags=cv2.INTER_NEAREST)
                # Post-process: assign maximum value where rotated cell is 0.
                rotated_cell[rotated_cell == 0] = np.max(rotated_cell)

                # Update the corresponding tile in the global map.
                # Use the precomputed coordinates.
                self.update_cell(rotated_cell, (cell_coords_x[i, j], cell_coords_y[i, j]))

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
    # Load homography matrix (assuming default path; adjust as needed)
    H = HomographyMatrix().get_homography_matrix()
    parser = argparse.ArgumentParser(description="Generate and update a global BEV cost map.")
    parser.add_argument("-b", type=str, required=True, help="Bag directory with synchronized pickle file and models subfolder inside.")
    parser.add_argument("-sim", "--use-sim-kmeans", action="store_true", help="Use the simulation k-means model instead of the default one")
    args = parser.parse_args()

    # Validate bag directory
    bag_path = args.b
    if not os.path.exists(bag_path) or not os.path.isdir(bag_path):
        raise FileNotFoundError(f"Bag path does not exist or is not a directory: {bag_path}")

    # Search for synced pickle file
    synced_pkl = [file for file in os.listdir(bag_path) if file.endswith("_synced.pkl")]
    if len(synced_pkl) != 1:
        raise FileNotFoundError(f"Exactly one '*_synced.pkl' file expected in '{bag_path}', found {len(synced_pkl)}")
    synced_pkl_path = os.path.join(bag_path, synced_pkl[0])

    # Search for .pt file in models subfolder
    models_dir = os.path.join(bag_path, "models")
    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        raise FileNotFoundError(f"'models' subfolder not found in '{bag_path}'")
    
    terrain_rep_files = [file for file in os.listdir(models_dir) if file.endswith("_terrain_rep.pt")]
    if len(terrain_rep_files) != 1:
        raise FileNotFoundError(f"Exactly one '*_terrain_rep.pt' file expected in '{models_dir}', found {len(terrain_rep_files)}")
    viz_encoder_path = os.path.join(models_dir, terrain_rep_files[0])

    # Define k-means model paths
    DEFAULT_KMEANS_PATH = "scripts/clusters/kmeans_model.pkl"
    SIM_KMEANS_PATH = "scripts/clusters_sim/sim_kmeans_model.pkl"

    PREFERENCES = {
        # Black: 0, White: 255
        0: 175,      #Cluster 0: Dark concrete, leaves, grass
        1: 0,      #Cluster 1: Smooth concrete
        2: 50,      #Cluster 2: Dark bricks, some grass
        3: 0,      #Cluster 3: Aggregate concrete, smooth concrete
        4: 225,      #Cluster 4: Grass
        5: 225,      # Cluster 5: Leaves, Grass
        #6: 0      # Cluster 6: Smooth concrete
    }

    SIM_PREFERENCES = {
        # Black: 0, White: 255
        0: 50,      #Cluster 0: Bricks
        1: 225,      #Cluster 1: Grass
        2: 0,      #Cluster 2: Pavement, bricks
        3: 175,      #Cluster 3: Mulch
        4: 225,      #Cluster 4: Grass
        5: 225,      # Cluster 5: Grass
        #6: 50      # Cluster 6: Smooth concrete
    }

    # Select k-means model and preferences based on -sim argument
    if args.use_sim_kmeans:
        kmeans_path = SIM_KMEANS_PATH
        preferences = SIM_PREFERENCES
    else:
        kmeans_path = DEFAULT_KMEANS_PATH
        preferences = PREFERENCES
    
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(f"K-means model not found at '{kmeans_path}'")

    # Debugging output (optional)
    print(f"Using synced pickle file: {synced_pkl_path}")
    print(f"Using visual encoder: {viz_encoder_path}")
    print(f"Using k-means model: {kmeans_path}")
    print(f"Using preferences: {preferences}")

    PREFERNCES = {
        # Black: 0, White: 255
        0: 175,      #Cluster 0: Dark concrete, leaves, grass
        1: 0,      #Cluster 1: Smooth concrete
        2: 50,      #Cluster 2: Dark bricks, some grass
        3: 0,      #Cluster 3: Aggregate concrete, smooth concrete
        4: 225,      #Cluster 4: Grass
        5: 225,      # Cluster 5: Leaves, Grass
        #6: 0      # Cluster 6: Smooth concrete
    }

    SIM_PREFERENCES = {
        # Black: 0, White: 255
        0: 50,      #Cluster 0: Bricks
        1: 225,      #Cluster 1: Grass
        2: 0,      #Cluster 2: Pavement, bricks
        3: 175,      #Cluster 3: Mulch
        4: 225,      #Cluster 4: Grass
        5: 225,      # Cluster 5: Grass
        #6: 50      # Cluster 6: Smooth concrete
    }

    bev_costmap = BEVCostmap(viz_encoder_path, kmeans_path, preferences)
    global_costmap = GlobalCostmap(tile_size=2560, cell_size=128, meters_per_pixel=1/(557))

    # Process each timestep
    robot_data = RobotDataAtTimestep(synced_pkl_path)
    odom_prev = np.eye(4, dtype=np.float32)  # Initial odometry is identity

    for timestep in tqdm(range(0, robot_data.getNTimesteps()), desc="Processing costmaps"):
        cur_img = robot_data.getImageAtTimestep(timestep)
        odom_cur = robot_data.getOdomAtTimestep(timestep)

        bev_img = plot_BEV_full(cur_img, H,patch_size=(128,128))
        costmap = bev_costmap.BEV_to_costmap(bev_img, 128)
        visualize_cost = bev_costmap.visualize_costmap(costmap, 128)

        global_costmap.process_odometry_and_update(visualize_cost, odom_cur)

        if timestep % 5 == 0:
            updated_costmap = global_costmap.get_combined_costmap()
            cv2.namedWindow("Global Cost Map", cv2.WINDOW_NORMAL)
            cv2.imshow("Global Cost Map", updated_costmap)
            cv2.waitKey(10)

    final_map = global_costmap.get_combined_costmap()
    viewer = MapViewer(final_map)
    viewer.show_map()
    viewer.save_full_map(filename="full_costmap.png")