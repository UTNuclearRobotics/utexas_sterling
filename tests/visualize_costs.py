"""
Load a trained model and visualize the costs associated with
the terrain patches from a camera feed saved as a video.
"""

import argparse
import os
import pickle
from termcolor import cprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import sys

from sterling.models import VisualEncoderEfficientModel, CostNet


class CostVisualizer:
    def __init__(self, model_path):
        self.model_path = model_path

        visual_encoder = VisualEncoderEfficientModel(latent_size=64)
        cost_net = CostNet(latent_size=64)

        # Load trained model
        self.model = nn.Sequential(visual_encoder, cost_net)
        model_state_dict = torch.load(os.path.join(model_path, "cost_model.pt"))
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        self.model.cuda()

        cprint("Model loaded and initialized", "green")

    def forward(self, bev_image, stride=1):
        """
        Args:
            bev_image (torch.Tensor): [C, H, W]
            stride (int): stride of the sliding window
        """
        patches = bev_image.unfold(1, 64, stride).unfold(2, 64, stride)  # Adjust dimensions as necessary
        patches = patches.contiguous().view(-1, 3, 64, 64)

        with torch.no_grad():
            cost = self.model(patches.cuda())

        # Find patches with sum of pixels == 0 and set their cost to 0
        idx = torch.sum(patches, dim=(1, 2, 3)) == 0
        cost[idx] = 0

        # Determine the number of patches horizontally and vertically
        num_patches_h = (bev_image.shape[1] - 64) // stride + 1
        num_patches_w = (bev_image.shape[2] - 64) // stride + 1

        costm = cost.view(num_patches_h, num_patches_w)  # [num_patches_h, num_patches_w]
        cost = F.interpolate(
            costm.unsqueeze(0).unsqueeze(0), size=(bev_image.shape[1], bev_image.shape[2]), mode="nearest"
        )
        # costm = cost.view((704 - 64) // stride + 1, (1472 - 64) // stride + 1)
        # cost = F.interpolate(costm.unsqueeze(0).unsqueeze(0), size=(704, 1472), mode="nearest")
        return cost.squeeze().cpu().numpy()


def test_cost_model(pkl_path, model_path, max_val=6.0, stride=64):
    cost_viz = CostVisualizer(model_path)
    output_video_path = os.path.join(model_path, pkl_path.split("/")[-1].split(".pkl")[0] + ".mp4")

    # Load processed data from pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Verify key in data
    bev_images = data["bev_imgs"]
    if bev_images is None:
        cprint("The pickle file does not contain 'bev_imgs'.", "red")
        return

    # Get number of messages
    num_images = len(bev_images)
    cprint(f"Number of BEV images to process: {num_images}", "green")

    # Initialize video writer
    frame_height, frame_width = bev_images[0].shape[:2]
    video_width = frame_width * 2  # BEV image + Cost map side by side
    video_height = frame_height
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"MP4V"), 20, (video_width, video_height))

    try:
        for bev_image in tqdm(bev_images, desc="Processing BEV images"):
            # Preprocess image
            bev_tensor = torch.from_numpy(bev_image.transpose(2, 0, 1).astype(np.float32) / 255.0)  # [C, H, W]

            # Compute cost
            cost = cost_viz.forward(bev_tensor, stride=stride)  # [H, W]

            # Post-process cost map
            cost_map = np.clip((cost * 255.0 / max_val), 0, 255).astype(np.uint8)  # Normalize to [0, 255]
            cost_map_color = cv2.cvtColor(cost_map, cv2.COLOR_GRAY2RGB)
            cost_map_color = cv2.resize(cost_map_color, (frame_width, frame_height))

            # Stack BEV image and cost map
            stacked_img = np.hstack((bev_image, cost_map_color))  # [H, W*2, C]
            stacked_img = cv2.cvtColor(stacked_img, cv2.COLOR_RGB2BGR)

            # Write to video
            video_writer.write(stacked_img)

        video_writer.release()
        cprint(f"Video saved to: {output_video_path}", "green")
    except KeyboardInterrupt:
        cprint("Video generation interrupted. Saving video...", "yellow")
        video_writer.release()
        cprint(f"Video saved to: {output_video_path}", "green")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BEV image costs from a ROS2 bag.")
    parser.add_argument(
        "--pkl_path", "-p", type=str, required=True, help="Path to the pickle file (processed ROS bag)."
    )
    parser.add_argument(
        "--model_path", "-m", type=str, required=True, help="Path to the folder containing the model weights."
    )
    parser.add_argument("--max_val", type=float, default=6.0, help="Maximum value for cost normalization.")
    parser.add_argument("--stride", type=int, default=64, help="Stride of the sliding window.")
    args = parser.parse_args()

    test_cost_model(
        pkl_path=args.pkl_path,
        model_path=args.model_path,
        max_val=args.max_val,
        stride=args.stride,
    )
