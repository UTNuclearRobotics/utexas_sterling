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

class GlobalCostmap:
    def __init__(self, map_size=(10000, 10000), cell_size=32, meters_per_pixel=0.0038, expansion_buffer=1280):
        """
        Initializes a large global cost map.

        Args:
            map_size: (height, width) of the global map in pixels.
            cell_size: Size of each cell in the costmap.
            meters_per_pixel: Conversion factor from real-world meters to pixels.
            expansion_buffer: Extra space to add when expanding, reducing frequent expansions.
        """
        self.map_size = list(map_size)  # Allow dynamic resizing
        self.cell_size = cell_size
        self.meters_per_pixel = meters_per_pixel
        self.expansion_buffer = expansion_buffer  # Extra buffer for expanding

        # Initialize costmap, mask, and depth buffer
        self.global_costmap = np.full(self.map_size, 255, dtype=np.uint8)  # Max cost
        self.global_mask = np.zeros(self.map_size, dtype=bool)  # Track updates
        self.z_buffer = np.copy(self.global_costmap)  # Depth buffer
        self.global_origin_x = self.map_size[1] // 2
        self.global_origin_y = self.map_size[0] // 2

    def expand_canvas(self, new_x, new_y):
        """
        Expands the global costmap only when an update is within 128 pixels of the boundary.
        When expanding, it increases by 1280 pixels instead of small increments.
        Prevents visual jumps when expanding upward by adjusting the origin instead of shifting the image.
        
        Args:
            new_x: X coordinate (in pixels) of the incoming cell.
            new_y: Y coordinate (in pixels) of the incoming cell.
        """
        h, w = self.map_size
        buffer_threshold = 128  # Start expansion when within 128 pixels
        expansion_amount = 1280  # Expand by 1280 pixels at a time

        expand_left = new_x < buffer_threshold
        expand_top = new_y < buffer_threshold
        expand_right = new_x + self.cell_size > w - buffer_threshold
        expand_bottom = new_y + self.cell_size > h - buffer_threshold

        # Calculate required new dimensions
        new_w = w
        new_h = h
        shift_x = 0
        shift_y = 0

        if expand_left:
            shift_x = expansion_amount  # Move everything right
            new_w += expansion_amount

        if expand_top:
            shift_y = expansion_amount  # Instead of shifting image, shift the reference point
            new_h += expansion_amount

        if expand_right:
            new_w += expansion_amount  # Expand to the right

        if expand_bottom:
            new_h += expansion_amount  # Expand downward

        # Check if expansion is required
        if new_w > w or new_h > h:
            print(f"Expanding map from ({h}, {w}) to ({new_h}, {new_w}) with shift ({shift_y}, {shift_x})")

            # Create new expanded arrays with the same default values
            new_costmap = np.full((new_h, new_w), 255, dtype=np.uint8)
            new_mask = np.zeros((new_h, new_w), dtype=bool)
            new_z_buffer = np.full((new_h, new_w), 255, dtype=np.uint8)

            # Copy old data into the new expanded arrays at the correct position
            # If expanding LEFT or BOTTOM, shift old map in new space
            y_offset = shift_y if expand_top else 0
            x_offset = shift_x if expand_left else 0

            new_costmap[y_offset:y_offset+h, x_offset:x_offset+w] = self.global_costmap
            new_mask[y_offset:y_offset+h, x_offset:x_offset+w] = self.global_mask
            new_z_buffer[y_offset:y_offset+h, x_offset:x_offset+w] = self.z_buffer

            # Assign new expanded arrays
            self.global_costmap = new_costmap
            self.global_mask = new_mask
            self.z_buffer = new_z_buffer
            self.map_size = [new_h, new_w]

            # Instead of shifting the whole costmap, adjust the global origin reference
            self.global_origin_x += shift_x
            self.global_origin_y += shift_y


    def update_cell(self, cell, global_position):
        """
        Updates a 64x64 cell in the global costmap without overwriting previous information.
        
        Args:
            cell: 2D numpy array (64x64) representing a costmap cell.
            global_position: (x, y) pixel coordinates in the global costmap.
        """
        y_offset, x_offset = global_position

        # Expand the canvas with buffer before updating
        self.expand_canvas(x_offset, y_offset)

        # Ensure cell remains within valid bounds after expansion
        x_end = min(x_offset + self.cell_size, self.map_size[1])
        y_end = min(y_offset + self.cell_size, self.map_size[0])
        x_start = max(x_offset, 0)
        y_start = max(y_offset, 0)

        # Extract the region of interest (ROI) from the global costmap
        global_roi = self.global_costmap[y_start:y_end, x_start:x_end]
        new_cell_roi = cell[:(y_end - y_start), :(x_end - x_start)]

        # Mask out regions that have already been written (not 255)
        empty_mask = global_roi == 255  # True where global_costmap is still uninitialized

        # Apply updates only to previously uninitialized cells
        update_mask = empty_mask & (new_cell_roi != 255)  # Ensure new data is valid

        # Apply updates
        global_roi[update_mask] = new_cell_roi[update_mask]


    def get_global_costmap(self):
        """Returns the global costmap."""
        return self.global_costmap


def process_odometry_and_update(global_map, local_costmap, odom_matrix):
    """
    Extracts (x, y) position and rotation from odometry and updates the global costmap.

    Args:
        global_map: Instance of GlobalBEVCostmap.
        local_costmap: 2D numpy array (costmap).
        odom_matrix: 4x4 numpy array representing the robot's odometry.
    """
    # Extract translation (negated for correct costmap alignment)
    t_x, t_y = odom_matrix[1, 3], odom_matrix[0, 3]

    # Convert meters to pixels
    x_pixel = int(t_x / global_map.meters_per_pixel)
    y_pixel = int(t_y / global_map.meters_per_pixel)

    # Shift origin to center of global map
    global_center_x = global_map.global_costmap.shape[0] // 2
    global_center_y = global_map.global_costmap.shape[1] // 2

    x_pixel += global_center_x
    y_pixel += global_center_y

    # Correct yaw extraction
    theta = np.arctan2(odom_matrix[1, 0], odom_matrix[0, 0])  
    theta_degrees = np.degrees(theta)

    # Convert local costmap to grayscale if needed
    if len(local_costmap.shape) == 3:
        local_costmap = cv2.cvtColor(local_costmap, cv2.COLOR_BGR2GRAY)

    # Get costmap dimensions
    h, w = local_costmap.shape
    num_cells_x = w // global_map.cell_size
    num_cells_y = h // global_map.cell_size

    # Use correct rotation matrix (Negate theta and fix Y-axis flipping)
    rotation_matrix = np.array([
        [np.cos(-theta), np.sin(-theta)],  # Fix Y flipping
        [-np.sin(-theta), np.cos(-theta)]
    ])

    # Loop through each 64x64 cell
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # Extract 64x64 cell
            cell = local_costmap[i * global_map.cell_size: (i + 1) * global_map.cell_size,
                                 j * global_map.cell_size: (j + 1) * global_map.cell_size]

            # Correct Local Position Calculation (Remove extra negation on Y)
            local_pos = np.array([
                -(j - num_cells_x // 2) * global_map.cell_size,  
                (i - num_cells_y // 2) * global_map.cell_size  # Remove negation
            ])

            # Apply Corrected Rotation
            rotated_pos = rotation_matrix @ local_pos
            rotated_pos[1] *= -1

            # Compute the global position
            cell_x = global_map.global_costmap.shape[0] - (x_pixel + int(rotated_pos[0]))  # Flipping X
            cell_y = y_pixel + int(rotated_pos[1])

            # Snap to 64-pixel grid
            cell_x = (cell_x // global_map.cell_size) * global_map.cell_size
            cell_y = (cell_y // global_map.cell_size) * global_map.cell_size

            # Rotate the cell before adding (Ensures local structure is aligned)
            rotated_cell = rotate_cell_bottom_center(cell, theta_degrees)

            # Update the global map with the rotated cell
            global_map.update_cell(rotated_cell, (cell_x, cell_y))

def rotate_cell_bottom_center(cell, angle):
    """
    Rotates a 64x64 cell around the bottom center while keeping alignment.

    Args:
        cell: 2D numpy array (grayscale) representing a single cell.
        angle: Rotation angle in degrees.

    Returns:
        Rotated cell.
    """
    h, w = cell.shape
    # Set rotation center to the bottom center of the cell
    center = (w // 2, h)  # Bottom center (x, y)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation with nearest-neighbor to avoid blurring
    rotated_cell = cv2.warpAffine(cell, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    
    # Remove tiny gaps by setting all near-empty pixels to the closest non-empty value
    rotated_cell[rotated_cell == 0] = np.max(rotated_cell)

    return rotated_cell


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

    for timestep in tqdm(range(0, 4000), desc="Processing costmaps"):
        cur_img = robot_data.getImageAtTimestep(timestep)
        odom_cur = robot_data.getOdomAtTimestep(timestep)
        robot_x, robot_y = -odom_cur[1, 3], -odom_cur[0, 3]

        bev_img = cv2.warpPerspective(cur_img, H, dsize)
        costmap = bev_costmap.BEV_to_costmap(bev_img, 32)
        visualize_cost = bev_costmap.visualize_costmap(costmap, 32)

        process_odometry_and_update(global_costmap, visualize_cost, odom_cur)

        if timestep % 5 == 0:
            updated_costmap = global_costmap.get_global_costmap()
            cv2.namedWindow("Global Cost Map", cv2.WINDOW_NORMAL)
            cv2.imshow("Global Cost Map", updated_costmap)
            cv2.waitKey(10)

    final_map = global_costmap.get_global_costmap()
    cv2.namedWindow("Global Cost Map", cv2.WINDOW_NORMAL)
    cv2.imshow("Global Cost Map", updated_costmap)
    key = cv2.waitKey(0)

    # Check if the pressed key is 'q'
    if key == ord('q'):
        cv2.destroyAllWindows()
    #viewer = MapViewer(final_map)
    #viewer.show_map()
    #viewer.save_full_map()