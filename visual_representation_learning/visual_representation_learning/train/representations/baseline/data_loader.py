import copy
import os
import pickle
import random

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from termcolor import cprint
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    """
    Custom Dataset class for loading and transforming data from a pickle file.

    Attributes:
        pickle_file_path (str): Path to the pickle file containing the data.
        data (dict): Loaded data from the pickle file.
        label (str): Label extracted from the file path.
        transforms (albumentations.Compose): Transformations to be applied to the images.
    """

    def __init__(self, pickle_file_path, train=False):
        """
        Initializes the dataset by loading data from the pickle file and setting up transformations.

        Args:
            pickle_file_path (str): Path to the pickle file containing the data.
            train (bool): Flag indicating whether the dataset is for training or validation.
        """
        cprint("Loading data from {} (train={})".format(pickle_file_path, train))
        self.data = pickle.load(open(pickle_file_path, "rb"))
        self.label = pickle_file_path.split("/")[-1]
        # self.label = pickle_file_path.split("/")[-2] # TODO: Folder name significance?

        if train:
            self.transforms = A.Compose(
                [
                    A.Rotate(
                        always_apply=False,
                        p=1.0,
                        limit=(-180, 180),
                        interpolation=3,
                        border_mode=0,
                        value=(0, 0, 0),
                        mask_value=None,
                    ),
                    A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                    A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 9)),
                    A.RandomBrightnessContrast(
                        always_apply=False, p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)
                    ),
                    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                    ToTensorV2(),
                ]
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data["patches"])

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing two image patches, inertial data, and the label.
        """
        patch_1 = copy.deepcopy(random.sample(self.data["patches"][idx], 1)[0])
        patch_1 = cv2.resize(patch_1, (128, 128))
        patch_1 = self.transforms(image=patch_1)["image"]

        patch_2 = copy.deepcopy(random.sample(self.data["patches"][idx], 1)[0])
        patch_2 = cv2.resize(patch_2, (128, 128))
        patch_2 = self.transforms(image=patch_2)["image"]

        inertial_data = self.data["imu"][idx]
        # inertial_data = np.expand_dims(inertial_data, axis=0) # TODO: Why expand_dims?

        return patch_1, patch_2, inertial_data, self.label


class MyDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and processing datasets.

    Attributes:
        data_config_path (str): Path to the YAML configuration file containing dataset paths.
        batch_size (int): Batch size for data loading.
        train_dataset (ConcatDataset): Concatenated training dataset.
        val_dataset (ConcatDataset): Concatenated validation dataset.
        inertial_stat (dict): Dictionary containing inertial data statistics (max and min values).
    """

    def __init__(self, data_config_path, batch_size=32):
        """
        Initializes the DataLoader.
        """
        super(MyDataModule, self).__init__()
        self.data_config_path = data_config_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        """
        Sets up the datasets and computes or loads inertial data statistics.
        """
        print("Setup method called with stage:", stage)
        try:
            with open(self.data_config_path) as file:
                data_config = yaml.safe_load(file)

            train_data_paths = data_config["train"]
            val_data_paths = data_config["val"]

            self.train_dataset = ConcatDataset([MyDataset(file, train=True) for file in train_data_paths])
            self.val_dataset = ConcatDataset([MyDataset(file, train=False) for file in val_data_paths])

            inertial_statistics_file_path = ".".join(self.data_config_path.split(".")[:-1]) + "_istat.yaml"
            if not os.path.exists(inertial_statistics_file_path):
                print(inertial_statistics_file_path, "path does not exist.")

                tmp = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
                inertial_list = []

                for _, _, i, _ in tqdm(tmp, desc="Computing inertial statistics"):
                    inertial_list.append(i)

                inertial_list = torch.cat(inertial_list, dim=0).reshape((-1, 1200)).numpy()
                max_inertial, min_inertial = inertial_list.max(axis=0), inertial_list.min(axis=0)
                self.inertial_stat = {"max": max_inertial.tolist(), "min": min_inertial.tolist()}

                print("Inertial data statistics have been created.")

                with open(inertial_statistics_file_path, "w") as file:
                    yaml.dump(self.inertial_stat, file)
            else:
                print(inertial_statistics_file_path, "path exists. Loading statistics.")
                with open(inertial_statistics_file_path, "r") as file:
                    tmp = yaml.full_load(file)
                    max_inertial = np.array(tmp["max"], dtype=np.float32)
                    min_inertial = np.array(tmp["min"], dtype=np.float32)
                    self.inertial_stat = {"max": max_inertial, "min": min_inertial}

                print("Inertial data statistics have been loaded.")

            print("Length of training dataset:", len(self.train_dataset))
            print("Length of validation dataset:", len(self.val_dataset))
        except Exception as e:
            print("Error in setup method:", e)
            raise

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False,
        )
