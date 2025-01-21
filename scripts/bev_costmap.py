import argparse
import os
import pickle
import torch

import cv2
import numpy as np
from homography_from_chessboard import HomographyFromChessboardImage
from robot_data_at_timestep import RobotDataAtTimestep
from termcolor import cprint
from tqdm import tqdm
from cluster import Cluster
import joblib


# GCD of 1280 and 720: 1,2,4,5,8,10,16,20,40,80
CELL_SIZE = 40


class BEVCostmap:
    """
    An overview of the cost inference process for local planning at deployment.
    """

    def __init__(self, synced_pkl_path, homography_cb, model_path, preferences):
        self.synced_pkl_path = synced_pkl_path
        self.homography_cb = homography_cb
        self.model_path = model_path
        self.preferences = preferences

        self.processed_imgs = {"bev": [], "cost": []}
        self.SAVE_PATH = "/".join(synced_pkl_path.split("/")[:-1])

        self.cluster = Cluster(data_pkl_path=os.path.join(script_dir, "../datasets/ahg_courtyard_1_vicreg.pkl"),
                               model_path=os.path.join(script_dir, "../models/vis_rep.pt"))

    def process_images(self):
        robot_data = RobotDataAtTimestep(self.synced_pkl_path)

        with tqdm(total=robot_data.getNTimesteps(), desc="Processing images to cost BEV") as pbar:
            for i in range(robot_data.getNTimesteps()):
                # Get the image
                img = robot_data.getImageAtTimestep(i)

                # Convert image to BEV
                bev_img = self.image_to_BEV(img)

                # TODO: Verify/calculate cell size to be GCD of image dimensions

                # Convert BEV image to costmap
                costmap = self.BEV_to_costmap(bev_img, CELL_SIZE)

                # Visualize costmap
                costmap_img = self.visualize_costmap(costmap, CELL_SIZE)

                self.processed_imgs["bev"].append(bev_img)
                self.processed_imgs["cost"].append(costmap_img)
                pbar.update(1)

    def image_to_BEV(self, img):
        """
        Plots the bird's-eye view (BEV) image using optimized rectangle parameters.
        """
        bev_img = self.homography_cb.plot_BEV_full(img)

        return bev_img

    def BEV_to_costmap(self, bev_img, cell_size):
        """
        Convert BEV image to costmap.
        Splits the BEV image into a grid and assigns a cost to each cell.
        Returns:
            costmap: A 2D numpy array representing the costmap.
        """
        height, width = bev_img.shape[:2]
        num_cells_y, num_cells_x = height // cell_size, width // cell_size


        # Initialize the costmap with zeros
        costmap = np.zeros((num_cells_y, num_cells_x), dtype=np.uint8)

        def calculate_cell_cost(cell):

            # Convert cell to required format
            if len(cell.shape) == 2:  # Grayscale
                cell = np.expand_dims(cell, axis=0)  # [H, W] -> [1, H, W]
            elif len(cell.shape) == 3 and cell.shape[2] == 3:  # RGB
                cell = np.transpose(cell, (2, 0, 1))  # [H, W, C] -> [C, H, W]

            # Convert cell to torch tensor
            cell_tensor = torch.tensor(cell, dtype=torch.float32)

            cluster_label = self.cluster.predict_cluster(cell_tensor, self.model_path)
            # Match image cell to a cluster using "predict_cluster" function
            # Assign grayscale value (0, 255) based on preference
            cost_value = self.preferences.get(cluster_label, 0)
            return cost_value

        # Iterate over each cell in the grid
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                # Extract the cell from the BEV image
                cell = bev_img[i * cell_size : (i + 1) * cell_size, j * cell_size : (j + 1) * cell_size]
                # Check if the cell is already black
                if np.all(cell == 0):  # All pixel values in the cell are black
                    cost = 255  # Assign maximum cost
                else:
                    # Calculate the cost for the cell using the clustering method
                    cost = calculate_cell_cost(cell)

                # Assign the cost to the corresponding cell in the costmap
                costmap[i, j] = cost

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
    # TODO: Add args for other files you need
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
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    image_dir = script_dir + "/homography/"
    image_file = "raw_image.jpg"
    chessboard_calibration_image = cv2.imread(os.path.join(image_dir, image_file))
    homography_cb = HomographyFromChessboardImage(chessboard_calibration_image, 8, 6)

    # TODO: Hardcode files so that you can test the script
    model_path = "scripts/clusters/kmeans_model.pkl"
    scaler_path = "scripts/clusters/scaler.pkl"
    preferences = {
        # Black: 0, White: 255
        0: 200,      #Cluster 0: Grass
        1: 0,        #Cluster 1: Smooth Concrete
        2: 100,      #Cluster 2: Aggregate Concrete
        3: 0,     #Cluster 3: Smooth concrete
        4: 0       #Cluster 4: Smooth Concrete

    }
    bev_costmap = BEVCostmap(
        synced_pkl_path=synced_pkl_path, homography_cb=homography_cb, model_path=model_path, preferences=preferences
    )
    bev_costmap.process_images()
    bev_costmap.save_data()
