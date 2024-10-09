#!/usr/bin/env python3
"""
process_rosbag.py

ROS node that listens to camera, IMU, and odometry information, processes the data, 
and saves it into a pickle file. 

The main functionalities include:
1. Subscribing to ROS topics to receive IMU, camera, and odometry data.
2. Synchronizing the received data.
3. Processing the data to extract patches from images based on odometry information.
4. Saving the processed data into a pickle file.
"""