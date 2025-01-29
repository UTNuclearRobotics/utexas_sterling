# Project Setup

## Prerequisites

- **ROS 2 Humble**: Ensure you have ROS 2 Humble installed on your system. Follow the official [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html) for your operating system.
- **Python 3.8+**: Ensure you have Python 3.8 or higher installed.

## ceres_python_bindings
Third-party bindings to allow the use of ceres-solver with Python.

```sh
git clone https://github.com/Edwinem/ceres_python_bindings.git
cd ceres_python_bindings
git submodule init
git submodule update
pip install .
```


## Python Virtual Environment

Run the scripts from inside a virtual environment.

1. Create the virtual environment and install dependencies:

```sh
chmod +x setup.sh
./setup.sh
```

2. To enter the virtual environment with ROS sourced:

```sh
source /opt/ros/humble/setup.bash
source venv/bin/activate
```