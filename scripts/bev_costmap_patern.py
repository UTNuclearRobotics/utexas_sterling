import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import joblib
from homography_from_chessboard import HomographyFromChessboardImage
from homography_params import get_homography_params
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
    An overview of the cost inference process for local planning at deployment using trained preference predictor.
    """

    def __init__(self, viz_encoder_path, uvis_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load visual encoder model weights
        self.sterling = SterlingPaternRepresentation(self.device).to(self.device)
        if not os.path.exists(viz_encoder_path):
            raise FileNotFoundError(f"Visual encoder file not found at: {viz_encoder_path}")
        self.sterling.visual_encoder.load_state_dict(torch.load(viz_encoder_path, weights_only=True, map_location=self.device))
        print(f"Loaded visual encoder from {viz_encoder_path}")

        # Load preference predictor (uvis)
        self.uvis = nn.Sequential(
            nn.Linear(128, 64),  # Assuming latent_size=128 from train_patern.py
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        ).to(self.device)
        if not os.path.exists(uvis_path):
            raise FileNotFoundError(f"Preference predictor file not found at: {uvis_path}")
        self.uvis.load_state_dict(torch.load(uvis_path, weights_only=True, map_location=self.device))
        print(f"Loaded preference predictor from {uvis_path}")

        self.processed_imgs = {"bev": [], "cost": []}

    def predict_preferences(self, cells):
        """Predict preferences for a batch of cells using the trained uvis model."""
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=torch.float32, device=self.device)

        if len(cells.shape) == 4:  # [B, C, H, W]
            pass  
        elif len(cells.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            cells = cells.unsqueeze(0)
        
        self.sterling.eval()
        self.uvis.eval()
        with torch.no_grad():
            representation_vectors = self.sterling.visual_encoder(cells)
            preferences = self.uvis(representation_vectors)  # [B, 1]
            # Scale to 0-255 range for costmap compatibility
            preferences = torch.sigmoid(preferences) * 255  # Assuming uvis outputs need normalization
            #preferences = preferences * 255
            costs = preferences.squeeze(-1).cpu().numpy().astype(np.uint8)
        return costs

    def BEV_to_costmap(self, bev_img, cell_size):
        """Convert BEV image to costmap while automatically marking consistent black areas."""
        height, width = bev_img.shape[:2]
        num_cells_y, num_cells_x = height // cell_size, width // cell_size

        effective_height = num_cells_y * cell_size
        effective_width = num_cells_x * cell_size
        bev_img = bev_img[:effective_height, :effective_width]

        costmap = np.empty((num_cells_y, num_cells_x), dtype=np.uint8)

        mask = np.zeros((height, width), dtype=np.uint8)
        triangle_left = np.array([[0, height], [0, 3 * height // 4], [width // 4, height]], dtype=np.int32)
        triangle_right = np.array([[width, height], [width, 3 * height // 4], [width - width // 4, height]], dtype=np.int32)
        cv2.fillPoly(mask, [triangle_left, triangle_right], 255)
        mask = mask[:effective_height, :effective_width]

        black_cells = (mask.reshape(num_cells_y, cell_size, num_cells_x, cell_size)
                            .max(axis=(1, 3)) == 255)

        channels = bev_img.shape[2]
        cell_shape = (num_cells_y, num_cells_x, cell_size, cell_size, channels)
        cell_strides = (bev_img.strides[0] * cell_size,
                        bev_img.strides[1] * cell_size,
                        bev_img.strides[0],
                        bev_img.strides[1],
                        bev_img.strides[2])
        cells = np.lib.stride_tricks.as_strided(bev_img, shape=cell_shape, strides=cell_strides)
        cells = cells.transpose(0, 1, 4, 2, 3)  # [num_cells_y, num_cells_x, channels, cell_size, cell_size]

        valid_cells = cells[~black_cells]

        if valid_cells.size:
            costs = self.predict_preferences(valid_cells)
        else:
            costs = np.empty((0,), dtype=np.uint8)

        costmap[black_cells] = 255
        costmap[~black_cells] = costs
        return costmap

    @staticmethod
    def visualize_costmap(costmap, cell_size):
        return cv2.cvtColor(cv2.resize(costmap, None, fx=cell_size, fy=cell_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    def save_data(self):
        frame_size = self.processed_imgs["bev"][0].shape[1], self.processed_imgs["bev"][0].shape[0]
        fps = 10
        combined_frame_size = (frame_size[0], frame_size[1]*2)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_save_path = os.path.join(self.SAVE_PATH, "costmap.mp4")
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, combined_frame_size)

        for i in tqdm(range(len(self.processed_imgs["cost"])), desc="Writing video"):
            img_cost = self.processed_imgs["cost"][i]
            img_BEV = self.processed_imgs["bev"][i]
            combined_frame = cv2.vconcat([img_cost, img_BEV])
            video_writer.write(combined_frame)

        video_writer.release()
        print(f"Video saved successfully: {video_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get BEV cost visual using trained preference predictor.")
    parser.add_argument("-b", type=str, required=True, help="Bag directory with synchronized pickle file inside.")
    args = parser.parse_args()

    bag_path = args.b
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    synced_pkl = [file for file in os.listdir(bag_path) if file.endswith("_synced.pkl")]
    if len(synced_pkl) != 1:
        raise FileNotFoundError(f"Synced pickle file not found in: {bag_path}")
    synced_pkl_path = os.path.join(bag_path, synced_pkl[0])

    H = get_homography_params().homography_matrix()
    robot_data = RobotDataAtTimestep(synced_pkl_path)

    # Paths to trained models from train_patern.py
    viz_encoder_path = "bags/agh_courtyard_2/models/fvis.pt"
    uvis_path = "bags/agh_courtyard_2/models/uvis.pt"

    bev_costmap = BEVCostmap(viz_encoder_path, uvis_path)

    for timestep in tqdm(range(1400, robot_data.getNTimesteps()), desc="Processing patches at timesteps"):
        cur_img = robot_data.getImageAtTimestep(timestep)
        cur_rt = robot_data.getOdomAtTimestep(timestep)
        bev_img = plot_BEV_full(cur_img, H, patch_size=(128, 128))
        costmap = bev_costmap.BEV_to_costmap(bev_img, 128)
        visualize = bev_costmap.visualize_costmap(costmap, 128)

        combined_frame = cv2.vconcat([visualize, bev_img])
        cv2.namedWindow("Cost Map", cv2.WINDOW_NORMAL)
        cv2.imshow("Cost Map", combined_frame)
        cv2.waitKey(10)

    cv2.destroyAllWindows()
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
