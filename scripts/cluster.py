import os
from terrain_dataset import TerrainDataset
from sterling_representation import SterlingRepresentation
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from util import load_dataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "../models/")
    model_filename = "vis_rep.pt"
    model_file = model_dir + model_filename

    data_pkl = load_dataset()

    # Contains N_SAMPLES of N_PATCHES each
    patches = data_pkl["patches"]
    
    # Create dataset and dataloader
    dataset = TerrainDataset(patches)
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)

    # Initialize model
    model = SterlingRepresentation(device).to(device)    

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    train_model()

    # dataloader = DataLoader(dataset, batch_size=8192, shuffle=False)
    # batch = next(iter(dataloader))
    # patch1, _ = batch

    # # patch1 = torch.cat([data[0] for data in dataloader])
    # # patch2 = torch.cat([data[1] for data in dataloader])
    # print("long_tensor.shape:  ", patch1.shape)
    # # patch1 = patch1[0]
    # patch1 = patch1.to(device)
    # representation_vectors = model.visual_encoder(patch1)