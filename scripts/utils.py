import os
import pickle

import torch
from termcolor import cprint

script_dir = os.path.dirname(os.path.abspath(__file__))


def load_dataset():
    dataset_dir = os.path.join(script_dir, "../datasets/")
    dataset_file = "nrg_ahg_courtyard.pkl"
    dataset_path = dataset_dir + dataset_file

    with open(dataset_path, "rb") as file:
        data_pkl = pickle.load(file)

    return data_pkl


def load_model(model):
    model_dir = os.path.join(script_dir, "../models/")
    model_file = "vis_rep.pt"
    model_path = model_dir + model_file

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        cprint("Existing model weights loaded successfully", "green")
    else:
        cprint("Existing model weights not found", "yellow")
    
    return model_path
