from setuptools import setup, find_packages
import os

setup(
    name="sterling",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",  # Match ROS Humble / Ubuntu 22.04 Python version
    install_requires=[
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
    },
    zip_safe=False,  # Recommended for packages with scripts/data
)