from setuptools import find_packages, setup


package_name = "visual_representation_learning"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/" + package_name + "/config", ["config/rosbag.yaml", "config/dataset.yaml"]),
        (
            "share/" + package_name + "/launch",
            ["launch/process_rosbag.launch.py"],
        ),
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "albumentations",
        "opencv-python",
        "pytorch-lightning",
        "tensorboard",
        "tensorflow",
        "torch",
        "torchinfo",
        "torchvision",
    ],
    zip_safe=True,
    maintainer="nchan",
    maintainer_email="nick.chan@utexas.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            ###
            ### Preprocessing
            ###
            "process_rosbag = visual_representation_learning.process_rosbag:main",
            ###
            ### Train
            ###
            "train_autoencoder_representations = visual_representation_learning.train.representations.baseline.train_autoencoder_representations:main",
            "train_sterling_representations = visual_representation_learning.train.representations.train_sterling_representations:main",
            "train_barlow_twins_representations = visual_representation_learning.train.representations.baseline.train_barlow_twins_representations:main",
            ###
            ### Utils
            ###
            # Convert PyTorch model to TorchScript JIT
            "convert_pt_jit = visual_representation_learning.train.convert_pt_jit:main",
            ###
            ### Visualize
            ###
            # Visualize PyTorch model convolution layers
            "visualize_models = visual_representation_learning.train.models:visualize_models",
        ],
    },
)
