"""
pkl_metadata.py

Visualize the size of data from PKL files.
"""
import argparse
import os
import pickle

from ament_index_python import get_package_share_directory

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize data from a PKL file.")
    parser.add_argument(
        "--file_path", type=str, default=os.path.join(ros_ws_dir, "datasets"), help="Path to the PKL files"
    )
    args = parser.parse_args()

    # Recursively find all pickle files in the dataset directory and its subdirectories
    pkl_files = []
    for root, _, files in os.walk(args.file_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    print(f"Found {len(pkl_files)} PKL files")

    for pkl_file in pkl_files:
        # Load data from the pkl file
        with open(pkl_file, "rb") as file:
            data = pickle.load(file)

        # Output the length of each array in the data
        print(pkl_file)
        for label, array in data.items():
            print(f"{label}: {len(array)}")
