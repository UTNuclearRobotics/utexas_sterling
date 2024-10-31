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
        "efficientnet_pytorch",
        "kneed",
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
            # Preprocessing
            "process_rosbag = visual_representation_learning.process_rosbag:main",
            # Train
            "train_representations = visual_representation_learning.train.representations.train_representations:main",
            "train_costs = visual_representation_learning.train.costs.train_costs:main",
            # Visualize
            "visualize_models_representations = visual_representation_learning.train.representations.models:visualize_models",
            "plot_costs = visual_representation_learning.visualize.plot_cost:main",
            "pkl_metadata = visual_representation_learning.visualize.pkl_metadata:main",
            # Utils
            "convert_pt_jit = visual_representation_learning.train.convert_pt_jit:main",
        ],
    },
)
