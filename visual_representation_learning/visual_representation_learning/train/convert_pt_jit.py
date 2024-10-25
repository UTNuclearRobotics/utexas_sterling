# TODO: Untested

"""
This script loads the CostNet and converts it to a TorchScript model.
"""

import argparse
import os
import sys

from ament_index_python.packages import get_package_share_directory

import torch
import torch.nn as nn
from visual_representation_learning.train.representations.models import CostNet, VisualEncoderModel

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))


def main():
    print('PyTorch version: ', torch.__version__)

    pt_file_default_path=os.path.join(ros_ws_dir, "torch", "models", "test.pt"),
    
    # pt_file_path is the first argument, else use the default path
    pt_file_path = sys.argv[1] if len(sys.argv) > 1 else pt_file_default_path
    
    # Verify if the pt_file_path exists
    if not os.path.isfile(pt_file_path):
        raise FileNotFoundError(f"The specified .pt file does not exist: {pt_file_path}")

    jit_file_save_path =pt_file_path.replace('.pt', '.jit')

    # load the model
    visual_encoder = VisualEncoderModel(latent_size=128)
    cost_net = CostNet(latent_size=128)
    model = nn.Sequential(visual_encoder, cost_net)
    model.load_state_dict(torch.load(pt_file_path))

    class WrappedModel(nn.Module):
        def __init__(self, model):
            super(WrappedModel, self).__init__()
            self.model = model
        def forward(self, x):
            return self.model(x.float() / 255.0)
            
    model = WrappedModel(model)

    # convert to TorchScript
    model = torch.jit.script(model)
    # save the TorchScript model
    torch.jit.save(model, jit_file_save_path)
    print('Saved TorchScript model to: ', jit_file_save_path)