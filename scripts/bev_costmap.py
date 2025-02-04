import argparse
import os
import pickle
import torch
import cv2
import numpy as np
import joblib
from homography_from_chessboard import HomographyFromChessboardImage
from robot_data_at_timestep import RobotDataAtTimestep
from termcolor import cprint
from tqdm import tqdm
from cluster import Cluster
import joblib
from sklearn.cluster import KMeans
from train_representation import SterlingRepresentation


# GCD of 1280 and 720: 1,2,4,5,8,10,16,20,40,80
CELL_SIZE = 40


class BEVCostmap:
    """
    An overview of the cost inference process for local planning at deployment.
    """

    def __init__(self, viz_encoder_path, kmeans_path, scaler_path, preferences):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load visual encoder model weights
        self.sterling = SterlingRepresentation(device).to(device)
        if not os.path.exists(viz_encoder_path):
            raise FileNotFoundError(f"Model file not found at: {viz_encoder_path}")
        self.sterling.load_state_dict(torch.load(viz_encoder_path, weights_only=True))

        # Load K-means model
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"K-means model not found at {kmeans_path}")
        self.kmeans = joblib.load(kmeans_path)

        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        self.preferences = preferences
        self.processed_imgs = {"bev": [], "cost": []}
        #self.SAVE_PATH = "/".join(synced_pkl_path.split("/")[:-1])

    def predict_clusters(self, cells):
        """Predict clusters for a batch of cells."""
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=torch.float32)

        if len(cells.shape) == 4:  # [B, C, H, W]
            pass  
        elif len(cells.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            cells = cells.unsqueeze(0)
        
        with torch.no_grad():
            representations = self.sterling.encode_single_patch(cells)

        representations_np = representations.detach().cpu().numpy()
        scaled_representations = self.scaler.transform(representations_np)
        cluster_labels = self.kmeans.predict(scaled_representations)

        return cluster_labels

    def calculate_cell_costs(self, cells):
        """Batch process cell costs."""
        cluster_labels = self.predict_clusters(cells)
        costs = [self.preferences.get(label, 0) for label in cluster_labels]
        return costs

    def BEV_to_costmap(self, bev_img, cell_size):
        """Convert BEV image to costmap using batch processing."""
        height, width = bev_img.shape[:2]
        num_cells_y, num_cells_x = height // cell_size, width // cell_size
        costmap = np.zeros((num_cells_y, num_cells_x), dtype=np.uint8)

        # Precompute all cells as a batch
        cells = [
            bev_img[i * cell_size : (i + 1) * cell_size, j * cell_size : (j + 1) * cell_size]
            for i in range(num_cells_y)
            for j in range(num_cells_x)
        ]

        # Convert cells to tensors in batch
        processed_cells = []
        for cell in cells:
            if np.all(cell == 0):  
                processed_cells.append(None)  # Mark black cells separately
            else:
                if len(cell.shape) == 2:  # Grayscale
                    cell = np.expand_dims(cell, axis=0)
                elif len(cell.shape) == 3 and cell.shape[2] == 3:  # RGB
                    cell = np.transpose(cell, (2, 0, 1))
                processed_cells.append(cell)

        # Convert to tensor batch
        valid_cells = [cell for cell in processed_cells if cell is not None]
        if valid_cells:
            valid_cells = np.stack(valid_cells, axis=0)
            costs = self.calculate_cell_costs(valid_cells)

        # Assign costs back to the costmap
        idx = 0
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                if processed_cells[i * num_cells_x + j] is None:
                    costmap[i, j] = 255  # Assign maximum cost to black cells
                else:
                    costmap[i, j] = costs[idx]
                    idx += 1

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
        height, width = costmap.shape
        img_height = height * cell_size
        img_width = width * cell_size

        # Create an empty grayscale image
        costmap_img = np.zeros((img_height, img_width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                cell_value = costmap[i, j]
                cv2.rectangle(
                    costmap_img,
                    (j * cell_size, i * cell_size),
                    ((j + 1) * cell_size, (i + 1) * cell_size),
                    int(cell_value),
                    thickness=cv2.FILLED,
                )

        color_costmap_img = cv2.cvtColor(costmap_img, cv2.COLOR_GRAY2BGR)
        return color_costmap_img

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
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Load the image
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    image = cv2.imread(os.path.join(image_dir, image_file))

    chessboard_homography = HomographyFromChessboardImage(image, 8, 6)
    #H = np.linalg.inv(chessboard_homography.H)  # get_homography_image_to_model()
    H, dsize,_ = chessboard_homography.plot_BEV_full(image)
    robot_data = RobotDataAtTimestep(synced_pkl_path)

    viz_encoder_path = "models/vis_rep.pt"
    kmeans_path = "scripts/clusters/kmeans_model.pkl"
    scaler_path = "scripts/clusters/scaler.pkl"

    preferences = {
        # Black: 0, White: 255
        0: 50,      #Cluster 0: Aggregate Concrete
        1: 225,      #Cluster 1: Metal thing
        2: 0,      #Cluster 2: Smooth concrete
        3: 225,      #Cluster 3: Grass
        4: 100,      #Cluster 4: Aggregate concrete, leaves
        5: 225,      # Cluster 5: Grass
        6: 0      # Cluster 6: Smooth concrete
    }

    bev_costmap = BEVCostmap(viz_encoder_path, kmeans_path, scaler_path, preferences)

    for timestep in tqdm(range(0, 4000), desc="Processing patches at timesteps"):
        cur_img = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)
        bev_img = cv2.warpPerspective(cur_img, H, dsize)  # Create BEV image
        costmap = bev_costmap.BEV_to_costmap(bev_img, 64)
        visualize = bev_costmap.visualize_costmap(costmap, 64)
        combined_frame = cv2.vconcat([visualize, bev_img])
        cv2.namedWindow("Cost Map", cv2.WINDOW_NORMAL)
        cv2.imshow("Cost Map", combined_frame)
        cv2.waitKey(0)

