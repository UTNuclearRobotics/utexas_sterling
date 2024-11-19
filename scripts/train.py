import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import pickle

matplotlib.use('TkAgg')

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, '../datasets/')
dataset_pkl = "nrg_ahg_courtyard.pkl"
dataset_file = dataset_dir + dataset_pkl

with open(dataset_file, 'rb') as file:
    data_pkl = pickle.load(file)

patches = data_pkl['patches']
print(len(patches[0][0][0]))
for index, patch in enumerate(patches):
    plt.imshow(patch[0])
    plt.axis('off')  # Turn off the axes for a cleaner image
    print(index)
    plt.pause(0.01)   # Pause for 0.2 seconds

print(data_pkl.keys())