import glob
import os
import pickle

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import yaml
from ament_index_python.packages import get_package_share_directory
from scipy.signal import periodogram
from termcolor import cprint
from tqdm import tqdm

import torch

from torch.utils.data import ConcatDataset, DataLoader, Dataset

torch.multiprocessing.set_sharing_strategy("file_system")  # https://github.com/pytorch/pytorch/issues/11201

package_share_directory = get_package_share_directory("visual_representation_learning")
ros_ws_dir = os.path.abspath(os.path.join(package_share_directory, "..", "..", "..", ".."))

IMU_TOPIC_RATE = 200.0


class TerrainDataset(Dataset):
    def __init__(self, pickle_file_path, data_stats=None, img_augment=False):
        cprint("Loading data from {}".format(pickle_file_path))
        self.pickle_files_path = pickle_file_path
        self.data = pickle.load(open(pickle_file_path, "rb"))
        self.label = pickle_file_path.split("/")[-2]
        self.data_stats = data_stats

        if data_stats is not None:
            self.min, self.max = data_stats["min"], data_stats["max"]
            self.mean, self.std = data_stats["mean"], data_stats["std"]

        if img_augment:
            cprint("Using image augmentations", "cyan")
            self.transforms = get_transforms()
        else:
            self.transforms = None

    def __len__(self):
        return len(self.data["patches"])

    def __getitem__(self, idx):
        imu = self.data["imu"][idx]
        PATCHES = self.data["patches"][idx]

        # Process IMU data
        # TODO: Reasoning behind this?
        # imu = imu[:, :-4]  # Exclude orientation data
        # imu = imu[:, [0, 1, 5]]  # Select angular_x, angular_y, linear_z
        # imu = periodogram(imu, fs=IMU_TOPIC_RATE, axis=0)[1]
        # imu = imu[-201:, :]  # Use the last 201 frequency components

        # Normalize IMU data if statistics are available
        if self.data_stats is not None:
            imu = (imu - self.min["imu"]) / (self.max["imu"] - self.min["imu"] + 1e-7)

        # Sample two random patches
        if len(PATCHES) > 1:
            patch_1_idx, patch_2_idx = np.random.choice(len(PATCHES), 2, replace=False)
        else:
            patch_1_idx = patch_2_idx = 0
        patch1, patch2 = PATCHES[patch_1_idx], PATCHES[patch_2_idx]

        # Convert BGR to RGB
        if not isinstance(patch1, np.ndarray):
            patch1 = np.array(patch1)
        if not isinstance(patch2, np.ndarray):
            patch2 = np.array(patch2)
        patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB)
        patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)

        # Apply transformations if available
        if self.transforms is not None:
            patch1 = self.transforms(image=patch1)["image"]
            patch2 = self.transforms(image=patch2)["image"]

        # Normalize image patches
        patch1 = np.asarray(patch1, dtype=np.float32) / 255.0
        patch2 = np.asarray(patch2, dtype=np.float32) / 255.0

        """
        Transpose the image patches from (H, W, C) to have channels first (C, H, W).
        This is often required by deep learning frameworks like PyTorch, 
        which expect image tensors to have the channels as the first dimension.
        
        C: Channels - This represents the number of color channels in the image (RGB)
        H: Height - Height of the image in pixels.
        W: Width - Width of the image in pixels.
        """
        patch1 = np.transpose(patch1, (2, 0, 1))
        patch2 = np.transpose(patch2, (2, 0, 1))

        return np.asarray(patch1), np.asarray(patch2), imu, self.label, idx


def get_transforms():
    return A.Compose(
        [
            A.Flip(always_apply=False, p=0.5),
            # A.CoarseDropout(always_apply=False, p=1.0, max_holes=5, max_height=16, max_width=16, min_holes=1, min_height=2, min_width=2, fill_value=(0, 0, 0), mask_fill_value=None),
            # A.AdvancedBlur(always_apply=False, p=0.1, blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
            A.ShiftScaleRotate(
                always_apply=False,
                p=0.75,
                shift_limit_x=(-0.1, 0.1),
                shift_limit_y=(-0.1, 0.1),
                scale_limit=(-0.1, 2.0),
                rotate_limit=(-21, 21),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                # mask_value=None,
                rotate_method="largest_box",
            ),
            A.Perspective(
                always_apply=False,
                p=0.5,
                scale=(0.025, 0.25),
                keep_size=1,
                pad_mode=0,
                pad_val=(0, 0, 0),
                mask_pad_val=0,
                fit_output=0,
                interpolation=3,
            ),
            # A.ISONoise(always_apply=False, p=0.5, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
            # A.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
            # A.ToGray(always_apply=False, p=0.5),
        ]
    )


# Create PyTorch Lightning data module
class SterlingDataModule(pl.LightningDataModule):
    def __init__(self, data_config_path, batch_size=64, num_workers=4):
        super().__init__()

        # Read the YAML configuration file
        cprint("Reading the configuration YAML file...", "yellow")
        self.data_config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)
        self.data_config_path = os.path.dirname(data_config_path)
        data_config_name = os.path.basename(data_config_path).split("/")[-1].split(".")[0]

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.mean = {}
        self.std = {}
        self.min = {}
        self.max = {}

        # Determine the path for data statistics pickle file
        self.data_statistics_pkl_path = os.path.join(self.data_config_path, f"{data_config_name}_statistics.pkl")
        if "all" in data_config_name:
            self.data_statistics_pkl_path = os.path.join(self.data_config_path, f"{data_config_name}_statistics_all.pkl")

        cprint(f"data_statistics_pkl_path: {self.data_statistics_pkl_path}", "cyan")

    @staticmethod
    def get_data_files(dirList):
        """
        Retrieve all pickle files from a list of directories.

        Args:
            dirList (list): A list of directory paths to search for pickle files.

        Returns:
            list: A list of paths to the found pickle files.
        """
        pickle_files = []

        for dir in dirList:
            # Get all pickle files in the current folder
            files = glob.glob(os.path.join(dir + "/*.pkl"))
            pickle_files.extend(files)

        return pickle_files

    def setup(self, stage=None):
        train_data_paths = SterlingDataModule.get_data_files(self.data_config["train"])
        val_data_paths = SterlingDataModule.get_data_files(self.data_config["val"])
        
        # Check if the data_statistics.pkl file exists
        if os.path.exists(self.data_statistics_pkl_path):
            cprint("Loading the mean and std from the data_statistics pickle file...", "yellow")
            data_statistics = pickle.load(open(self.data_statistics_pkl_path, "rb"))
        else:
            # Find the mean and std of the train dataset
            cprint("data_statistics pickle file not found!", "yellow")
            cprint("Finding the mean and std of the train dataset...", "yellow")

            # Create a temporary dataset and dataloader for calculating statistics
            # QUESTION: What is the point of this temporary dataset and dataloader? I see it gets fed into the TerrainDataset.
            tmp_dataset = ConcatDataset(
                [TerrainDataset(file) for file in train_data_paths]
            )
            self.tmp_dataloader = DataLoader(tmp_dataset, batch_size=128, num_workers=10, shuffle=True)
            cprint(f"The length of the tmp_dataloader is: {len(self.tmp_dataloader)}", "cyan")

            # Collect IMU data from the temporary dataloader
            imu_data = []
            for _, _, imu, _, _ in tqdm(self.tmp_dataloader):
                imu_data.append(imu.cpu().numpy())
            imu_data = np.concatenate(imu_data, axis=0)

            # Calculate mean, std, min, and max for IMU data
            self.mean["imu"], self.std["imu"] = np.mean(imu_data, axis=0), np.std(imu_data, axis=0)
            self.min["imu"], self.max["imu"] = np.min(imu_data, axis=0), np.max(imu_data, axis=0)

            # cprint(f"Mean: {self.mean}", "green")
            # cprint(f"Std: {self.std}", "green")
            # cprint(f"Min: {self.min}", "green")
            # cprint(f"Max: {self.max}", "green")

            # Save the calculated statistics to a pickle file
            cprint("Saving the mean, std, min, max to the data_statistics pickle file", "green")
            data_statistics = {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max}
            pickle.dump(data_statistics, open(self.data_statistics_pkl_path, "wb"))

        self.train_dataset = ConcatDataset(
            [TerrainDataset(file, data_stats=data_statistics, img_augment=True) for file in train_data_paths]
        )
        self.val_dataset = ConcatDataset([TerrainDataset(file, data_stats=data_statistics) for file in val_data_paths])

        # Log the length of the train and validation datasets
        cprint(f"Length of train dataset: {len(self.train_dataset)}", "green")
        cprint(f"Length of validation dataset: {len(self.val_dataset)}", "green")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False,
            pin_memory=True,
        )
