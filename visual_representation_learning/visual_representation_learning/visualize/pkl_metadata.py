import pickle
import argparse


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize data from a PKL file.")
    parser.add_argument("--file_path", type=str, help="Path to the PKL file")

    # Parse arguments
    args = parser.parse_args()

    # Load data from the pkl file
    with open(args.file_path, "rb") as file:
        data = pickle.load(file)

    # Output the length of each array in the data
    for label, array in data.items():
        print(f"{label}: {len(array)}")
    
    # Print the entire dictionary to see its contents
    print(data["patches"][10])
    print(len(data["patches"][10]))
    print(data["imu"][0])
    print(len(data["imu"][0]))
