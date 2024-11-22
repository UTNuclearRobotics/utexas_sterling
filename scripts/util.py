import os
import pickle

def load_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "../datasets/")
    dataset_pkl = "nrg_ahg_courtyard.pkl"
    dataset_file = dataset_dir + dataset_pkl
    with open(dataset_file, "rb") as file:
        data_pkl = pickle.load(file)
    return data_pkl

# def load_model():
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     model_dir = os.path.join(script_dir, "../models/")
#     model_filename = "vis_rep.pt"
#     model_file = model_dir + model_filename
