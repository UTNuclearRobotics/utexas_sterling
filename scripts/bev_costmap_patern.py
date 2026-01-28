import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from homography_params import get_homography_params
from robot_data_at_timestep import RobotDataAtTimestep
from termcolor import cprint
from tqdm import tqdm
from train_patern_minus import PaternPreAdaptation
from homography_utils import plot_BEV_full
import gc


# GCD of 1280 and 720: 1,2,4,5,8,10,16,20,40,80
CELL_SIZE = 128


class BEVCostmap:
    """
    Cost inference process for local planning at deployment using trained preference predictor.
    """

    def __init__(self, model_path, save_path=None, adapted=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path

        # Load visual encoder model weights
        self.model = PaternPreAdaptation(self.device).to(self.device)

        # Define the expected .pt files for each submodule
        if adapted:
            weight_files = {
                "visual_encoder": "fvis_adapted.pt",
                "proprioceptive_encoder": "fpro.pt",
                "uvis": "uvis_adapted.pt",
                "upro": "upro.pt",
                "cost_head": "cost_head_adapted.pt"
            }
        else:
            weight_files = {
                "visual_encoder": "fvis.pt",
                "proprioceptive_encoder": "fpro.pt",
                "uvis": "uvis.pt",
                "upro": "upro.pt",
                "cost_head": "cost_head.pt"
            }

        # Load weights for each submodule
        for submodule_name, file_name in weight_files.items():
            file_path = os.path.join(model_path, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Weight file for {submodule_name} not found at: {file_path}")
            
            # Load the state dict for the submodule
            state_dict = torch.load(file_path, weights_only=True, map_location=self.device)
            
            # Get the corresponding submodule from self.model
            submodule = getattr(self.model, submodule_name)
            submodule.load_state_dict(state_dict)
            print(f"Loaded {submodule_name} weights from {file_path}")

        # Set the model to evaluation mode
        self.model.eval()
        
        if self.save_path is not None:
            self.processed_imgs = {"bev": [], "cost": []}

    def predict_preferences(self, cells):
        """Predict preferences for a batch of cells using the trained uvis model."""
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=torch.float32, device=self.device)

        if len(cells.shape) == 4:  # [B, C, H, W]
            pass  
        elif len(cells.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            cells = cells.unsqueeze(0)
        
        with torch.no_grad():
            # Pass None for inertial data
            phi_vis, _, uvis_pred, _, final_cost = self.model(cells, inertial=None)

            #preferences = preferences * 255
            uvis_costs = uvis_pred.squeeze(-1).cpu().numpy().astype(np.uint8)
            final_costs = final_cost.squeeze(-1).cpu().numpy().astype(np.uint8)
            final_costs = np.clip(final_costs, 1, 100)
            return uvis_costs, final_costs

    def BEV_to_costmap(self, bev_img, cell_size):
        """Convert BEV image to costmap while automatically marking consistent black areas."""
        height, width = bev_img.shape[:2]
        num_cells_y, num_cells_x = height // cell_size, width // cell_size

        effective_height = num_cells_y * cell_size
        effective_width = num_cells_x * cell_size
        bev_img = bev_img[:effective_height, :effective_width]

        costmap = np.empty((num_cells_y, num_cells_x), dtype=np.uint8)

        mask = np.zeros((height, width), dtype=np.uint8)
        triangle_left = np.array([[0, height], [0, 1 * height // 4], [(width // 4)+256, height]], dtype=np.int32)
        triangle_right = np.array([[width, height], [width, 1 * height // 4], [(width - width // 4)-256, height]], dtype=np.int32)
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
                # Ensure valid_cells has shape [B, C, H, W]
                if len(valid_cells.shape) == 5 and valid_cells.shape[2] == 1:  # Grayscale
                    valid_cells = valid_cells.squeeze(2)  # [B, H, W]
                    valid_cells = np.stack([valid_cells] * 3, axis=1)  # [B, 3, H, W]
                uvis_cost, final_cost = self.predict_preferences(valid_cells)
                final_cost = (final_cost * (255.0 / 100.0)).astype(np.uint8)
        else:
            uvis_cost, final_cost = np.empty((0,), dtype=np.uint8)

        costmap[black_cells] = 255
        costmap[~black_cells] = final_cost

        #inv_costmap = 255 - costmap
        inv_costmap = costmap

        # Prepare costmap for video: resize to match bev_img dimensions and convert to 3 channels
        costmap_resized = cv2.resize(inv_costmap, (effective_width, effective_height), interpolation=cv2.INTER_NEAREST)
        costmap_3ch = np.stack([costmap_resized] * 3, axis=-1).astype(np.uint8)  # (H, W, 3)

        # Append to self.processed_imgs
        if self.save_path is not None:
            self.processed_imgs["bev"].append(bev_img)
            self.processed_imgs["cost"].append(costmap_3ch)

        return inv_costmap

    @staticmethod
    def visualize_costmap(costmap, cell_size):
        return cv2.cvtColor(cv2.resize(costmap, None, fx=cell_size, fy=cell_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    def save_data(self, video_writer, frame_count, frame_size=None):
        if not self.processed_imgs["bev"] or not self.processed_imgs["cost"]:
            print("Warning: No images to write in this segment.")
            return

        num_frames = len(self.processed_imgs["cost"])
        print(f"Writing {num_frames} frames to video")

        # Precompute expected size
        expected_size = (frame_size[0], frame_size[1] * 2) if frame_size else None

        # Disable tqdm for small batches to reduce overhead
        use_tqdm = num_frames > 100
        iterator = tqdm(range(num_frames), desc="Writing video segment") if use_tqdm else range(num_frames)

        # Process frames in batches to reduce overhead
        for i in iterator:
            img_cost = self.processed_imgs["cost"][i]
            img_BEV = self.processed_imgs["bev"][i]

            # Resize only once if necessary
            if frame_size:
                if img_BEV.shape[:2][::-1] != frame_size:  # Compare (width, height)
                    img_BEV = cv2.resize(img_BEV, frame_size, interpolation=cv2.INTER_AREA)
                if img_cost.shape[:2][::-1] != frame_size:
                    img_cost = cv2.resize(img_cost, frame_size, interpolation=cv2.INTER_AREA)

            # Ensure consistent dimensions (should be rare after initial resize)
            if img_cost.shape[1] != img_BEV.shape[1] or img_cost.shape[0] != img_BEV.shape[0]:
                img_cost = cv2.resize(img_cost, (img_BEV.shape[1], img_BEV.shape[0]), interpolation=cv2.INTER_AREA)

            # Normalize only if not already uint8 (batch this if possible)
            if img_BEV.dtype != np.uint8:
                img_BEV = img_BEV.astype(np.float32)
                img_min, img_max = img_BEV.min(), img_BEV.max()
                if img_max > img_min:  # Avoid division by zero
                    img_BEV = (img_BEV - img_min) / (img_max - img_min + 1e-6) * 255
                img_BEV = img_BEV.astype(np.uint8)

            if img_cost.dtype != np.uint8:
                img_cost = img_cost.astype(np.float32)
                img_min, img_max = img_cost.min(), img_cost.max()
                if img_max > img_min:
                    img_cost = (img_cost - img_min) / (img_max - img_min + 1e-6) * 255
                img_cost = img_cost.astype(np.uint8)

            img_BEV = cv2.cvtColor(img_BEV, cv2.COLOR_RGB2BGR)

            # Concatenate vertically
            combined_frame = cv2.vconcat([img_cost, img_BEV])

            # Ensure correct dimensions for VideoWriter
            if video_writer and expected_size:
                current_size = (combined_frame.shape[1], combined_frame.shape[0])  # (width, height)
                if current_size != expected_size:
                    print(f"Frame size mismatch: expected {expected_size}, got {current_size}")
                    combined_frame = cv2.resize(combined_frame, expected_size, interpolation=cv2.INTER_AREA)

                video_writer.write(combined_frame)

        # Clear processed_imgs after writing
        self.processed_imgs = {"bev": [], "cost": []}
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get BEV cost visual using trained preference predictor.")
    parser.add_argument("-m","-model_bag", type=str, required=True, help="Bag directory with model files inside.")
    parser.add_argument("-b","-synced_bag", type=str, required=True, help="Bag directory with synchronized HDF5 file inside.")
    parser.add_argument("-a","-use_adapted", type=bool, default=True, help="Use adapted models if True, else use preadapted model.")
    parser.add_argument("-v", "-save_vid", type=bool, default=False, help="Save video if True or play live if False.")
    args = parser.parse_args()

    model_path = args.m
    bag_path = args.b
    adapted = args.a
    save_vid = args.v
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    h5_files = [file for file in os.listdir(bag_path) if file.endswith("_synced.h5")]
    if len(h5_files) != 1:
        raise FileNotFoundError(f"Synchronized HDF5 file not found or multiple found in: {bag_path}")
    h5_file_path = os.path.join(bag_path, h5_files[0])

    H = get_homography_params().homography_matrix()
    robot_data = RobotDataAtTimestep(h5_file_path)  

    # Search for pre-trained weights
    models_dir = os.path.join(args.m, "models")
    save_path = bag_path if save_vid else None
    bev_costmap = BEVCostmap(models_dir, save_path=save_path, adapted=adapted)
    max_timesteps = robot_data.getNTimesteps()
    start_timestep = min(7000, max_timesteps)
    frame_count = 0
    video_writer = None
    frame_size = None

    try:
        for timestep in tqdm(range(start_timestep, max_timesteps), desc="Processing patches at timesteps"):
            try:
                cur_img = robot_data.getImageAtTimestep(timestep)
                if cur_img is None or cur_img.size == 0:
                    print(f"Warning: No image at timestep {timestep}, skipping.")
                    continue
                cur_rt = robot_data.getOdomAtTimestep(timestep)
                bev_img = plot_BEV_full(cur_img, H, patch_size=(128, 128))
                if bev_img is None or bev_img.size == 0:
                    print(f"Warning: Invalid BEV image at timestep {timestep}, skipping.")
                    continue
                costmap = bev_costmap.BEV_to_costmap(bev_img, 128)
                if costmap is None:
                    print(f"Warning: Costmap generation failed at timestep {timestep}, skipping.")
                    continue
                visualize = bev_costmap.visualize_costmap(costmap, 128)
                if visualize is None:
                    print(f"Warning: Visualization failed at timestep {timestep}, skipping.")
                    continue

                # Set frame size after first successful frame
                if frame_size is None:
                    frame_size = (bev_img.shape[1], bev_img.shape[0])  # (width, height)
                    combined_frame_size = (frame_size[0], frame_size[1] * 2)  # For vertical stacking

                    if save_vid:
                        video_save_path = os.path.join(args.synced_bag, "costmap.mp4")
                        print(f"Initializing VideoWriter with frame size {combined_frame_size}")
                        video_writer = cv2.VideoWriter(
                            video_save_path, 
                            cv2.VideoWriter_fourcc(*"mp4v"), 
                            10, 
                            combined_frame_size
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Failed to open VideoWriter for {video_save_path}")
                    else:
                        # Initialize OpenCV window for live display
                        cv2.namedWindow("BEV Costmap", cv2.WINDOW_NORMAL)
                        #cv2.resizeWindow("BEV Costmap", frame_size[0], frame_size[1] * 2)

                # Stack BEV and costmap vertically
                combined_img = np.vstack((bev_img, visualize))

                if save_vid:
                    # Write to video
                    video_writer.write(combined_img)
                else:
                    # Display live
                    cv2.imshow("BEV Costmap", combined_img)
                    # Wait for 1ms and check for 'q' key to exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User interrupted display.")
                        break

                frame_count += 1

                # Write and clear processed_imgs every 1000 frames (for save_vid=True)
                if save_vid and frame_count % 1000 == 0 and frame_count > 0:
                    print(f"Writing video segment at frame {frame_count}")
                    bev_costmap.save_data(video_writer, frame_count, frame_size)

            except Exception as e:
                print(f"Error at timestep {timestep}: {e}")
                continue
    except Exception as e:
        print(f"Critical error during loop: {e}. Saving video with processed frames.")
    finally:
        # Save any remaining frames (for save_vid=True)
        if save_vid and bev_costmap.processed_imgs["bev"] or bev_costmap.processed_imgs["cost"]:
            print(f"Saving remaining frames: {len(bev_costmap.processed_imgs['bev'])} BEV, {len(bev_costmap.processed_imgs['cost'])} Cost")
            bev_costmap.save_data(video_writer, frame_count, frame_size)
        
        # Release resources
        if save_vid and video_writer:
            video_writer.release()
        if not save_vid:
            cv2.destroyAllWindows()

        print(f"Total processed frames: {frame_count}")
        if save_vid and frame_count > 0:
            print(f"Video saved successfully: {video_save_path}")
            if os.path.exists(video_save_path):
                file_size = os.path.getsize(video_save_path)
                print(f"Video file size: {file_size / (1024 * 1024):.2f} MB")
            else:
                print(f"Video file does not exist at {video_save_path}")
        elif save_vid:
            print("No frames processed. Video not saved.")
        gc.collect()


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