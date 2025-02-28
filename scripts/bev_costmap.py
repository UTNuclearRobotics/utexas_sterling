import argparse
import os
import pickle
import torch
import cv2
import numpy as np
import joblib
from homography_from_chessboard import HomographyFromChessboardImage
from homography_matrix import HomographyMatrix
from robot_data_at_timestep import RobotDataAtTimestep
from termcolor import cprint
from tqdm import tqdm
from cluster import Cluster
import joblib
from sklearn.cluster import KMeans
from train_representation import SterlingPaternRepresentation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, normalize, RobustScaler
from homography_utils import plot_BEV_full


# GCD of 1280 and 720: 1,2,4,5,8,10,16,20,40,80
CELL_SIZE = 40


class BEVCostmap:
    """
    An overview of the cost inference process for local planning at deployment.
    """

    def __init__(self, viz_encoder_path, kmeans_path, preferences):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load visual encoder model weights
        self.sterling = SterlingPaternRepresentation(self.device).to(self.device)
        if not os.path.exists(viz_encoder_path):
            raise FileNotFoundError(f"Model file not found at: {viz_encoder_path}")
        self.sterling.load_state_dict(torch.load(viz_encoder_path, weights_only=True, map_location=torch.device(self.device)))

        # Load K-means model
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"K-means model not found at {kmeans_path}")
        self.kmeans = joblib.load(kmeans_path)

        # Load scaler
        #if not os.path.exists(scaler_path):
        #    raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        #self.scaler = joblib.load(scaler_path)

        self.preferences = preferences
        self.processed_imgs = {"bev": [], "cost": []}
        #self.SAVE_PATH = "/".join(synced_pkl_path.split("/")[:-1])

    def predict_clusters(self, cells):
        """Predict clusters for a batch of cells."""
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=torch.float32, device=self.device)

        if len(cells.shape) == 4:  # [B, C, H, W]
            pass  
        elif len(cells.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            cells = cells.unsqueeze(0)
        
        self.sterling.eval()
        with torch.no_grad():
            representation_vectors = self.sterling.visual_encoder(cells)
            # Ensure representation_vectors is on CPU
            representations_np = representation_vectors.cpu().numpy()
            representations_np = normalize(representations_np, axis=1, norm='l2')
        #scaled_representations = self.scaler.transform(representations_np)

        return self.kmeans.predict(representations_np)

    def calculate_cell_costs(self, cells):
        """Batch process cell costs."""
        cluster_labels = self.predict_clusters(cells)
        costs = [self.preferences.get(label, 0) for label in cluster_labels]
        return costs

    def BEV_to_costmap(self, bev_img, cell_size):
        """Convert BEV image to costmap while automatically marking consistent black areas."""
        height, width = bev_img.shape[:2]
        num_cells_y, num_cells_x = height // cell_size, width // cell_size

        # Determine effective dimensions that are multiples of cell_size.
        effective_height = num_cells_y * cell_size
        effective_width = num_cells_x * cell_size

        # Slice the image to the effective region.
        bev_img = bev_img[:effective_height, :effective_width]

        # Initialize costmap container.
        costmap = np.empty((num_cells_y, num_cells_x), dtype=np.uint8)

        # Create mask for black regions.
        # If these dimensions and cell_size remain constant, consider caching the following mask.
        mask = np.zeros((height, width), dtype=np.uint8)
        triangle_left = np.array([[0, height], [0, 3 * height // 4], [width // 4, height]], dtype=np.int32)
        triangle_right = np.array([[width, height], [width, 3 * height // 4], [width - width // 4, height]], dtype=np.int32)
        cv2.fillPoly(mask, [triangle_left, triangle_right], 255)
        mask = mask[:effective_height, :effective_width]

        # Reshape mask into cells and compute per-cell max to detect any black pixel.
        black_cells = (mask.reshape(num_cells_y, cell_size, num_cells_x, cell_size)
                            .max(axis=(1, 3)) == 255)

        # Use stride tricks to extract cell views without copying data.
        channels = bev_img.shape[2]
        cell_shape = (num_cells_y, num_cells_x, cell_size, cell_size, channels)
        cell_strides = (bev_img.strides[0] * cell_size,
                        bev_img.strides[1] * cell_size,
                        bev_img.strides[0],
                        bev_img.strides[1],
                        bev_img.strides[2])
        cells = np.lib.stride_tricks.as_strided(bev_img, shape=cell_shape, strides=cell_strides)
        # Rearrange to (num_cells_y, num_cells_x, channels, cell_size, cell_size)
        cells = cells.transpose(0, 1, 4, 2, 3)

        # Select only valid (non-black) cells.
        valid_cells = cells[~black_cells]

        # Calculate costs for valid cells in a single batch.
        if valid_cells.size:
            costs = self.calculate_cell_costs(valid_cells)
        else:
            costs = np.empty((0,), dtype=np.uint8)

        # Assemble costmap: assign maximum cost (255) to black cells and computed costs to others.
        costmap[black_cells] = 255
        costmap[~black_cells] = costs

        return costmap

    @staticmethod
    def visualize_costmap(costmap, cell_size):
        """
        Args:
            costmap: A 2D numpy array representing the costmap (values should be between 0 and 255).
            cell_size: Size of each cell in the costmap.
        Returns:
            Grayscale image of the costmap.
        """

        # Convert grayscale to BGR for consistency
        return cv2.cvtColor(cv2.resize(costmap, None, fx=cell_size, fy=cell_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    def save_data(self):
        # Initialize the video writer
        frame_size = self.processed_imgs["bev"][0].shape[1], self.processed_imgs["bev"][0].shape[0]
        fps = 10
        combined_frame_size = (frame_size[0], frame_size[1]*2)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_save_path = os.path.join(self.SAVE_PATH, "costmap.mp4")
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, combined_frame_size)

        for i in tqdm(range(len(self.processed_imgs["cost"])), desc="Writing video"):
            img_cost = self.processed_imgs["cost"][i]
            img_BEV = self.processed_imgs["bev"][i]

            # Combine frames side-by-side (horizontal concatenation)
            combined_frame = cv2.vconcat([img_cost, img_BEV])
            video_writer.write(combined_frame)

        video_writer.release()
        cprint(f"Video saved successfully: {video_save_path}", "green")


if __name__ == "__main__":
    """
    Notes:
        Looks for the synced pickle file in the bag directory (ends in "_synced.pkl").
        Uses the bag directory to save the video as "costmap.mp4".
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Get BEV cost visual.")
    parser.add_argument("-b", type=str, required=True, help="Bag directory with synchronzied pickle file inside.")
    args = parser.parse_args()

    # Check if the bag file exists
    bag_path = args.b
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    # Validate the sycned pickle file
    synced_pkl = [file for file in os.listdir(bag_path) if file.endswith("_synced.pkl")]
    if len(synced_pkl) != 1:
        raise FileNotFoundError(f"Synced pickle file not found in: {bag_path}")
    synced_pkl_path = os.path.join(bag_path, synced_pkl[0])

    # Get chessboard calibration image
if __name__ == "__main__":
    H = HomographyMatrix().get_homography_matrix()
    #H, dsize,_ = chessboard_homography.plot_BEV_full(image)
    robot_data = RobotDataAtTimestep(synced_pkl_path)

    viz_encoder_path = "bags/agh_courtyard_2/models/agh_courtyard_2_terrain_rep.pt"#"bags/ahg_courtyard_1/models/ahg_courtyard_1_terrain_rep.pt"
    kmeans_path = "scripts/clusters/kmeans_model.pkl"

    sim_encoder_path = "bags/panther_sim_recording_low_20250228_125700/models/panther_sim_recording_low_20250228_125700_terrain_rep.pt"#"bags/panther_recording_20250218_175547/models/panther_recording_20250218_175547_terrain_rep.pt"
    sim_kmeans_path = "scripts/clusters_sim/sim_kmeans_model.pkl"
    #scaler_path = "scripts/clusters_sim/sim_scaler.pkl"

    preferences = {
        # Black: 0, White: 255
        0: 175,      #Cluster 0: Dark concrete, leaves, grass
        1: 0,      #Cluster 1: Smooth concrete
        2: 50,      #Cluster 2: Dark bricks, some grass
        3: 0,      #Cluster 3: Aggregate concrete, smooth concrete
        4: 225,      #Cluster 4: Grass
        5: 225,      # Cluster 5: Leaves, Grass
        #6: 0      # Cluster 6: Smooth concrete
    }

    preferences_ahg_2 = {
        # Black: 0, White: 255
        0: 0,      #Cluster 0: Concrete
        1: 225,      #Cluster 1: Dark thing
        2: 50,      #Cluster 2: concrete??
        3: 225,      #Cluster 3: Dark thing
        4: 225,      #Cluster 4: Dark thing
        5: 225,      # Cluster 5: Dark thing
        6: 50,      # Cluster 6: Bricks
        7: 225,      # Cluster 7: Dark thing
        8: 175,      # Cluster 8: Everything, but most grass and leaves
        9: 0,      #Cluster 9: Concrete
    }

    sim_preferences = {
        # Black: 0, White: 255
        0: 225,      #Cluster 0: Grass
        1: 0,      #Cluster 1: Pavement
        2: 225,      #Cluster 2: Red
        3: 150,      #Cluster 3: Grass and pavement
        4: 225,      #Cluster 4: Grass and red
        #5: 225,      # Cluster 5: Grass
        #6: 50      # Cluster 6: Smooth concrete
    }

    bev_costmap = BEVCostmap(viz_encoder_path, kmeans_path, preferences_ahg_2)

    for timestep in tqdm(range(1400, robot_data.getNTimesteps()), desc="Processing patches at timesteps"):
        cur_img = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)
        bev_img = plot_BEV_full(cur_img, H,patch_size=(128,128))
        costmap = bev_costmap.BEV_to_costmap(bev_img, 128)
        visualize = bev_costmap.visualize_costmap(costmap, 128)

        combined_frame = cv2.vconcat([visualize, bev_img])
        cv2.namedWindow("Cost Map", cv2.WINDOW_NORMAL)
        cv2.imshow("Cost Map", combined_frame)
        cv2.waitKey(10)

# Building costmap from global map only
"""
    global_img = cv2.imread("full_map.png")
    costmap = bev_costmap.BEV_to_costmap(global_img, 64)
    visualize = bev_costmap.visualize_costmap(costmap, 64)
    cv2.namedWindow("Cost Map", cv2.WINDOW_NORMAL)
    cv2.imshow("Cost Map", visualize)
    cv2.waitKey(0)
    cv2.imwrite("costmap_from_global.png", visualize)
"""
