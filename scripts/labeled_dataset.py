import h5py
import argparse

def print_hdf5_shapes(hdf5_path):
    """
    Print the shapes of all datasets in an HDF5 file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file.
    """
    # Load HDF5 file
    with h5py.File(hdf5_path, 'r') as h5f:
        print(f"Inspecting HDF5 file: {hdf5_path}")
        print(f"Available datasets: {list(h5f.keys())}")
        print("-" * 50)
        
        # Print shapes for each dataset
        if 'patches' in h5f:
            patches_shape = h5f['patches'].shape
            print(f"Dataset 'patches': shape = {patches_shape}")
        
        if 'inertial' in h5f:
            inertial_shape = h5f['inertial'].shape
            print(f"Dataset 'inertial': shape = {inertial_shape}")
        else:
            print("Dataset 'inertial': Not present")
        
        if 'terrain_labels' in h5f:
            terrain_labels_shape = h5f['terrain_labels'].shape
            print(f"Dataset 'terrain_labels': shape = {terrain_labels_shape}")
        
        if 'preferences' in h5f:
            preferences_shape = h5f['preferences'].shape
            print(f"Dataset 'preferences': shape = {preferences_shape}")
        
        print("-" * 50)
        print("Shape inspection complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print shapes of datasets in an HDF5 file.")
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 file")
    
    args = parser.parse_args()
    
    print_hdf5_shapes(hdf5_path=args.hdf5_path)