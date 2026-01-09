#!/usr/bin/env python3
import argparse
import math
import os
import gc
import h5py

import cv2
import numpy as np
from homography_params import get_homography_params
from homography_utils import *
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils import *
from robot_data_at_timestep import RobotDataAtTimestep
import yaml

def ComputeVicRegData(H, K, plane_normal, plane_distance, robot_data, odom_offset, meters_per_pixel, history_size=10, patch_size=(128, 128), batch_size=1000, output_dir="", num_patches=1, lateral_pixel_shift=-128):
    print(f"Parameters: odom_offset={odom_offset}, meters_per_pixel={meters_per_pixel}, plane_distance={plane_distance}")
    n_timesteps = robot_data.getNTimesteps()

    # Define shifts for x and y directions
    shift_step_x = patch_size[0]  # Shift step for x
    shift_step_y = patch_size[1]  # Shift step for y
    x_shifts = np.arange(-num_patches, num_patches + 1) * shift_step_x + lateral_pixel_shift  # e.g., [-128, 0, 128]
    #y_shifts = [shift_step_y]  # Shift up by one row only (e.g., [128])
    y_shifts = [0]  # Shift up by one row only (e.g., [128])

    # Create all combinations of x and y shifts
    shift_combinations = [(x_shift, y_shift) for x_shift in x_shifts for y_shift in y_shifts]
    n_shifts = len(shift_combinations)  # Total number of shifts (e.g., 3 * 1 = 3)

    # Create transformation matrices for each shift combination
    T_shifts = np.tile(np.eye(3), (n_shifts, 1, 1))
    for i, (x_shift, y_shift) in enumerate(shift_combinations):
        T_shifts[i, 0, 2] = x_shift  # x translation
        T_shifts[i, 1, 2] = y_shift  # y translation
    H_shifted_all = np.matmul(T_shifts, H)

    #plane_distance = 1.0

    patch_corners = np.array([
        [0, 0], [patch_size[0], 0],
        [patch_size[0], patch_size[1]], [0, patch_size[1]]
    ], dtype=np.float32).reshape(-1, 1, 2)

    batch_files_created = []

    for batch_start in range(history_size, n_timesteps, batch_size):
        batch_end = min(batch_start + batch_size, n_timesteps)
        batch_patches = []
        batch_global_positions = []

        batch_start_with_history = max(0, batch_start - history_size)
        images = {t: robot_data.getImageAtTimestep(t) for t in range(batch_start_with_history, batch_end)}
        odometry = {t: robot_data.getOdomAtTimestep(t) for t in range(batch_start_with_history, batch_end)}

        for timestep in tqdm(range(batch_start, batch_end), desc=f"Processing batch {batch_start}-{batch_end}"):
            cur_image = images[timestep]
            cur_rt = odometry[timestep]
            R_cur, T_base = cur_rt[:3, :3], cur_rt[:3, 3]
            camera_offset_global = R_cur @ odom_offset
            T_cur = T_base + camera_offset_global

            patch_center_bev = np.array([patch_size[0]/2, patch_size[1]/2, 1])
            patch_center_bev_m = patch_center_bev[:2] * meters_per_pixel
            forward_vector = R_cur @ np.array([-1, 0, 0])
            patch_offset = forward_vector * patch_center_bev_m[0]
            patch_center_camera_3d = np.array([patch_offset[0], patch_center_bev_m[1], -1.0]) # -1.0 is camera height
            patch_center_global = R_cur @ patch_center_camera_3d + T_cur
            batch_global_positions.append(patch_center_global)
            if timestep < batch_start + 5:  # Print first 5 positions
                print(f"Timestep {timestep}: T_base={T_base[:2]}, T_cur={T_cur[:2]}")
                print("Odom Position:", T_base)
                print("Patch Position:", patch_center_global)

            cur_patches = np.array([
                cv2.warpPerspective(cur_image, H_shifted, dsize=patch_size)
                for H_shifted in H_shifted_all
            ])

            timestep_shift_patches = []
            for shift_idx, H_shifted in enumerate(H_shifted_all):
                timestep_patches = [cur_patches[shift_idx]]  # Current patch
                valid_past_timesteps = range(max(0, timestep - history_size + 1), timestep)
                past_images = [images[t] for t in valid_past_timesteps]
                past_rts = [odometry[t] for t in valid_past_timesteps]

                past_patches = []
                for past_image, past_rt in zip(past_images, past_rts):
                    R_past = past_rt[:3, :3]
                    T_past = past_rt[:3, 3] + (R_past @ odom_offset)
                    R_rel = R_cur.T @ R_past
                    T_rel = R_cur.T @ (T_past - T_cur)
                    H_past2cur = compute_homography_from_rt(K, R_rel, T_rel, plane_normal, plane_distance)
                    H_past2patch = H_shifted @ H_past2cur

                    transformed_corners = cv2.perspectiveTransform(patch_corners, H_past2cur)
                    bbox_coords = transformed_corners[:, 0, :].astype(int)
                    x_min, y_min = np.maximum(np.min(bbox_coords, axis=0), 0)
                    x_max, y_max = np.minimum(np.max(bbox_coords, axis=0), past_image.shape[1::-1])
                    img_height, img_width = past_image.shape[:2]
                    past_bbox = [x_min, y_min, x_max, y_max]
                    cur_bbox = [0, 0, patch_size[0], patch_size[1]]

                    if does_overlap(cur_bbox, past_bbox, img_width, img_height):
                        past_patch = cv2.warpPerspective(past_image, H_past2patch, dsize=patch_size)
                        past_patches.append(past_patch)

                timestep_patches.extend(past_patches)
                timestep_shift_patches.append(timestep_patches)

            batch_patches.append(timestep_shift_patches)

        batch_file = os.path.join(output_dir, f"batch_{batch_start}_{batch_end}.h5")
        with h5py.File(batch_file, "w") as h5f:
            for i, (shift_patches, global_pos) in enumerate(zip(batch_patches, batch_global_positions)):
                group = h5f.create_group(f"timestep_{i}")
                for shift_idx, timestep_patches in enumerate(shift_patches):
                    shift_group = group.create_group(f"shift_{shift_idx}")
                    for j, patch in enumerate(timestep_patches):
                        shift_group.create_dataset(
                            f"patch_{j}",
                            data=patch,
                            compression="gzip",
                            compression_opts=4   # 0–9, 4 is a good balance of speed vs compression
                        )
                group.create_dataset("global_position", data=global_pos)

        if not os.path.exists(batch_file) or os.path.getsize(batch_file) == 0:
            raise RuntimeError(f"Failed to save batch file: {batch_file}")
        
        print(f"Saved batch to {batch_file}")
        batch_files_created.append(batch_file)

        batch_patches = None
        batch_global_positions = None
        images = None
        odometry = None
        gc.collect()

    return batch_files_created

def does_overlap(cur_bbox, past_bbox, img_width, img_height):
    x_min_cur, y_min_cur, x_max_cur, y_max_cur = cur_bbox
    x_min_past, y_min_past, x_max_past, y_max_past = past_bbox

    if (x_min_cur < 0 or x_max_cur > img_width or y_min_cur < 0 or y_max_cur > img_height or
        x_min_past < 0 or x_max_past > img_width or y_min_past < 0 or y_max_past > img_height):
        return False

    return not (
        x_max_cur < x_min_past or x_max_past < x_min_cur or
        y_max_cur < y_min_past or y_max_past < y_min_cur
    )

def stitch_patches_in_grid(patches, grid_size=None, gap_size=10, gap_color=(255, 255, 255)):
    if not patches:
        return np.full((128, 128, 3), gap_color, dtype=np.uint8)
    
    patches = [patch[:, :, [2, 1, 0]] for patch in patches]

    if grid_size is None:
        num_patches = len(patches) - 1
        grid_cols = max(math.ceil(math.sqrt(num_patches)), 1)
        grid_rows = math.ceil(num_patches / grid_cols) if num_patches > 0 else 1
    else:
        grid_rows, grid_cols = grid_size

    patch_height, patch_width, _ = patches[0].shape
    grid_height = (grid_rows + 1) * patch_height + grid_rows * gap_size
    grid_width = max(grid_cols * patch_width + (grid_cols - 1) * gap_size, patch_width)
    canvas = np.full((int(grid_height), int(grid_width), 3), gap_color, dtype=np.uint8)

    canvas[:patch_height, :patch_width] = patches[0]

    for idx, patch in enumerate(patches[1:], start=1):
        row = (idx - 1) // grid_cols + 1
        col = (idx - 1) % grid_cols
        start_y = row * (patch_height + gap_size)
        start_x = col * (patch_width + gap_size)
        canvas[start_y : start_y + patch_height, start_x : start_x + patch_width] = patch

    return canvas

def validate_vicreg_data(vicreg_data_path):
    with h5py.File(vicreg_data_path, "r") as h5f:
        vicreg_data = []  # List of [shift][patch] for each timestep
        global_positions = []
        for timestep_key in sorted(h5f.keys(), key=lambda x: int(x.split('_')[1])):
            timestep_group = h5f[timestep_key]
            timestep_shift_patches = []
            for shift_key in sorted(timestep_group.keys(), key=lambda x: int(x.split('_')[1]) if 'shift' in x else -1):
                if 'shift' in shift_key:
                    shift_group = timestep_group[shift_key]
                    patches = [shift_group[patch_key][:] for patch_key in sorted(shift_group.keys(), key=lambda x: int(x.split('_')[1]))]
                    timestep_shift_patches.append(patches)
            vicreg_data.append(timestep_shift_patches)
            global_positions.append(timestep_group['global_position'][:] if 'global_position' in timestep_group else None)

    print(f"Number of timesteps: {len(vicreg_data)}")
    print(f"Number of shifts per timestep: {len(vicreg_data[0])}")
    print(f"Number of patches per shift: {len(vicreg_data[0][0])}")

    counter = 0  # Timestep index
    shift_idx = 0  # Shift index within timestep
    cv2.namedWindow("VICReg Data", cv2.WINDOW_NORMAL)

    while counter < len(vicreg_data):
        # Get patches for the current shift
        current_shift_patches = vicreg_data[counter][shift_idx]
        patch_images = stitch_patches_in_grid(current_shift_patches)
        cv2.imshow("VICReg Data", patch_images)

        key = cv2.waitKey(0)
        if key == 113:  # 'q' to quit
            break
        elif key == 82:  # Up arrow: next timestep, reset shift
            counter += 1
            shift_idx = 0
        else:  # Any other key: next shift, or next timestep if at last shift
            shift_idx += 1
            if shift_idx >= len(vicreg_data[counter]):  # If beyond last shift
                counter += 1
                shift_idx = 0

    cv2.destroyAllWindows()

def combine_batches_to_single_h5(batch_files, output_h5_path, chunk_size=500):
    batch_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    total_timesteps = 0

    with h5py.File(output_h5_path, "w") as h5f:
        for batch_file in batch_files:
            with h5py.File(batch_file, "r") as batch_h5f:
                batch_timesteps = len(batch_h5f.keys())
                print(f"Processing {batch_file} with {batch_timesteps} timesteps")

                for timestep_key in sorted(batch_h5f.keys(), key=lambda x: int(x.split('_')[1])):
                    timestep_group = batch_h5f[timestep_key]
                    group = h5f.create_group(f"timestep_{total_timesteps}")

                    # Copy shift subgroups
                    for shift_key in sorted(timestep_group.keys(), key=lambda x: int(x.split('_')[1]) if 'shift' in x else -1):
                        if 'shift' in shift_key:
                            shift_group_in = timestep_group[shift_key]
                            shift_group_out = group.create_group(shift_key)
                            for patch_key in sorted(shift_group_in.keys(), key=lambda x: int(x.split('_')[1])):
                                input_ds = shift_group_in[patch_key]

                                shift_group_out.create_dataset(
                                    patch_key,
                                    data=input_ds[:],
                                    compression=input_ds.compression,
                                    compression_opts=input_ds.compression_opts,
                                    chunks=input_ds.chunks
                                )

                    if 'global_position' in timestep_group:
                        group.create_dataset("global_position", data=timestep_group['global_position'][:])
                    else:
                        print(f"Warning: No global_position in {batch_file}, timestep {timestep_key}")

                    total_timesteps += 1

                gc.collect()

    print(f"Combined {total_timesteps} timesteps into {output_h5_path}")

def cleanup_batch_files(batch_files):
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            os.remove(batch_file)
            print(f"Deleted batch file: {batch_file}")
    print("All batch files cleaned up")

def calibrate_meters_per_pixel(robot_data, H, K, plane_normal, meters_per_pixel, plane_distance=1.0, patch_size=(128, 128), calibration_timestep=1500, visualize_steps=5, num_patches=1, lateral_pixel_shift=-128):
    """
    Compute cam_ground_offset and visualize all patch positions on the original image using provided meters_per_pixel.
    
    Args:
        robot_data: Object providing getOdomAtTimestep and getImageAtTimestep methods.
        H: Homography matrix.
        K: Camera intrinsic matrix.
        plane_normal: Normal vector of the plane.
        meters_per_pixel: Meters per pixel for scaling (e.g., 1/330 for 330 px/m).
        plane_distance: Distance to the plane (default: 1.0 meters).
        patch_size: Size of the patch in pixels (default: (128, 128)).
        calibration_timestep: Timestep to use for visualization (default: 1500).
        visualize_steps: Number of timesteps to visualize (default: 5).
    
    Returns:
        cam_ground_offset: Offset from camera to patch center in camera coordinates.
    """
    n_timesteps = robot_data.getNTimesteps()
    timestep = min(calibration_timestep, n_timesteps - 1)
    patch_corners_bev = np.array([
        [0, 0, 1],
        [patch_size[0], 0, 1],
        [patch_size[0], patch_size[1], 1],
        [0, patch_size[1], 1]
    ], dtype=np.float32)
    H_inv = np.linalg.inv(H)

    # Compute cam_ground_offset
    cur_image = robot_data.getImageAtTimestep(timestep)
    cur_rt = robot_data.getOdomAtTimestep(timestep)
    R_cur, T_base = cur_rt[:3, :3], cur_rt[:3, 3]
    robot_odom_position = cur_rt[:3, 3]

    patch_center_bev = np.array([patch_size[0]/2, patch_size[1]/2, 1])
    patch_center_image = H_inv @ patch_center_bev
    patch_center_image /= patch_center_image[2]
    u_target, v_target = patch_center_image[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_offset = (u_target - cx) / fx
    y_offset = -(v_target - cy) / fy
    cam_ground_offset = np.array([x_offset, 0.0, 0.0])

    print(f"Timestep {timestep}:")
    print(f"Offset from patch to camera local: {cam_ground_offset}")

    # Define shifts for visualization (same as ComputeVicRegData)
    shift_step_x = patch_size[0]
    shift_step_y = patch_size[1]
    x_shifts = np.arange(-num_patches, num_patches + 1) * shift_step_x + lateral_pixel_shift  # e.g., [-256, -128, 0, 128, 256]
    #y_shifts = [shift_step_y]  # Shift up by one row only (e.g., [128])
    y_shifts = [0]
    shift_combinations = [(x_shift, y_shift) for x_shift in x_shifts for y_shift in y_shifts]
    n_shifts = len(shift_combinations)

    # Create transformation matrices for each shift combination
    T_shifts = np.tile(np.eye(3), (n_shifts, 1, 1))
    for i, (x_shift, y_shift) in enumerate(shift_combinations):
        T_shifts[i, 0, 2] = x_shift
        T_shifts[i, 1, 2] = y_shift
    H_shifted_all = np.matmul(T_shifts, H)

    # Transformation matrix for reference patch (shifted up by one row)
    T_ref = np.eye(3)
    T_ref[0, 2] = lateral_pixel_shift      # Apply lateral correction
    #T_ref[1, 2] = shift_step_y  # Apply y-shift to reference patch
    H_ref = np.matmul(T_ref, H)
    H_ref_inv = np.linalg.inv(H_ref)

    # Patch center for reference patch (shifted up)
    patch_center_bev_ref = np.array([patch_size[0]/2, patch_size[1]/2, 1])
    patch_center_image = H_ref_inv @ patch_center_bev_ref
    patch_center_image /= patch_center_image[2]

    # Patch corners for visualization
    patch_corners = np.array([
        [0, 0], [patch_size[0], 0],
        [patch_size[0], patch_size[1]], [0, patch_size[1]]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Visualization phase
    cv2.namedWindow("Patch Visualization", cv2.WINDOW_NORMAL)
    for vis_timestep in range(timestep, min(timestep + visualize_steps, n_timesteps)):
        cur_image = robot_data.getImageAtTimestep(vis_timestep)
        cur_rt = robot_data.getOdomAtTimestep(vis_timestep)
        R_cur, T_base = cur_rt[:3, :3], cur_rt[:3, 3]
        camera_offset_global = R_cur @ cam_ground_offset
        T_cur = T_base + camera_offset_global
        robot_odom_position = cur_rt[:3, 3]

        # Patch center in BEV (pixels)
        patch_center_bev = np.array([patch_size[0]/2, patch_size[1]/2, 1])
        patch_center_bev_m = patch_center_bev[:2] * meters_per_pixel

        # Patch center with forward offset
        forward_vector = R_cur @ np.array([1, 0, 0])
        patch_offset = forward_vector * patch_center_bev_m[0]
        patch_center_camera_3d = np.array([patch_offset[0], patch_center_bev_m[1], -plane_distance])
        patch_center_global = R_cur @ patch_center_camera_3d + T_cur

        # Compute offset patch center using homography (to align with blue dot)
        offset_patch_center_bev = np.array([patch_size[0]/2, patch_size[1]/2, 1])
        offset_patch_center_image = H_ref_inv @ offset_patch_center_bev
        offset_patch_center_image /= offset_patch_center_image[2]
        offset_patch_center_image = offset_patch_center_image[:2]  # (u, v) in image coordinates

        distance_vector = patch_center_global - robot_odom_position

        # Draw on the image
        if len(cur_image.shape) == 2:
            vis_image = cv2.cvtColor(cur_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = cur_image.copy()

        # Draw all shifted patch boundaries and centers
        for shift_idx, H_shifted in enumerate(H_shifted_all):
            H_shifted_inv = np.linalg.inv(H_shifted)
            patch_corners_image = cv2.perspectiveTransform(patch_corners, H_shifted_inv)
            corners_xy = patch_corners_image[:, 0, :].astype(int)
            x_shift, y_shift = shift_combinations[shift_idx]
            color = (0, 255, 0) if x_shift == 0 and y_shift == shift_step_y else (255, 255, 0)  # Red for center, cyan for others
            cv2.polylines(vis_image, [corners_xy], isClosed=True, color=color, thickness=2)
            patch_center_image_shifted = H_shifted_inv @ patch_center_bev
            patch_center_image_shifted /= patch_center_image_shifted[2]
            center_xy = (int(patch_center_image_shifted[0]), int(patch_center_image_shifted[1]))
            cv2.circle(vis_image, center_xy, 5, color, -1)
            cv2.putText(vis_image, f"Shift ({x_shift}, {y_shift})", (center_xy[0] + 10, center_xy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw reference patch boundaries
        patch_corners_image = cv2.perspectiveTransform(patch_corners, H_ref_inv)
        corners_xy = patch_corners_image[:, 0, :].astype(int)
        cv2.polylines(vis_image, [corners_xy], isClosed=True, color=(0, 0, 255), thickness=2)

        # Draw reference patch center (blue dot)
        cv2.circle(vis_image, (int(patch_center_image[0]), int(patch_center_image[1])), 5, (0, 0, 255), -1)

        # Draw offset patch center (red dot)
        cv2.circle(vis_image, (int(offset_patch_center_image[0]), int(offset_patch_center_image[1])), 5, (255, 0, 0), -1)
        cv2.putText(vis_image, f"Dist: {np.linalg.norm(distance_vector):.2f}m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the image with patch boundaries
        cv2.imshow("Patch Visualization", vis_image)

        print(f"Timestep {vis_timestep}:")
        print(f"Robot odometry position (x, y, z): {robot_odom_position}")
        print(f"Global position of center patch (with offset): {patch_center_global}")
        print(f"Offset patch center image coords (red dot): {offset_patch_center_image}")
        for shift_idx, (x_shift, y_shift) in enumerate(shift_combinations):
            patch_center_image_shifted = np.linalg.inv(H_shifted_all[shift_idx]) @ patch_center_bev
            patch_center_image_shifted /= patch_center_image_shifted[2]
            print(f"Shift ({x_shift}, {y_shift}): Image coords {patch_center_image_shifted[:2]}")
        print(f"Reference patch (shifted up, blue dot): Image coords {patch_center_image[:2]}")

        cv2.waitKey(0)

    cam_ground_offset = np.array([np.linalg.norm(distance_vector), 0.0, 0.0])

    cv2.destroyAllWindows()
    return cam_ground_offset

if __name__ == "__main__":
    H = get_homography_params().homography_matrix()
    RT = get_homography_params().rigid_transform()
    plane_normal = get_homography_params().plane_norm()
    plane_distance = get_homography_params().plane_dist()
    K, _ = get_homography_params().camera_intrinsics()
    px_meter = get_homography_params().px_meter()

    parser = argparse.ArgumentParser(description="Preprocess data for VICReg.")
    parser.add_argument("-b", type=str, required=True, help="Bag directory with synchronized HDF5 file inside.")
    args = parser.parse_args()

    bag_path = args.b
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    synced_h5 = [file for file in os.listdir(bag_path) if file.endswith("_synced.h5")]
    if len(synced_h5) != 1:
        raise FileNotFoundError(f"Synced HDF5 file not found in: {bag_path}")
    synced_h5_path = os.path.join(bag_path, synced_h5[0])

    robot_data = RobotDataAtTimestep(synced_h5_path)
    save_path = "/".join(synced_h5_path.split("/")[:-1])
    vicreg_data_path = os.path.join(save_path, save_path.split("/")[-1] + "_vicreg.h5")
    
    camera_base_offset = [0.2286, 0, 0.5715]
    patch_size = 128
    num_patches = 2
    lateral_pixel_shift = patch_size // 2

    if os.path.exists(vicreg_data_path):
        print(f"Using existing HDF5 file: {vicreg_data_path}")
        validate_vicreg_data(vicreg_data_path)
    else:
        history_size = 10
        batch_size = 1000
        # Calibrate meters_per_pixel first
        cam_ground_offset = calibrate_meters_per_pixel(robot_data, H, K, plane_normal, meters_per_pixel=1/px_meter, plane_distance=1.0, patch_size=(patch_size, patch_size), calibration_timestep=100, visualize_steps=3, num_patches=num_patches, lateral_pixel_shift=lateral_pixel_shift)

        odom_offset = cam_ground_offset + camera_base_offset

        # Calculate total offset (odom_offset) and append to config.yaml
        magnitude = np.linalg.norm(cam_ground_offset)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'homography', 'config.yaml')
        data = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                data = yaml.safe_load(file) or {}
        data['cam_to_patch_offset'] = magnitude.tolist()
        with open(config_path, 'w') as file:
            yaml.dump(data, file)
        print(f"Total offset saved to {config_path}: {magnitude}")

        expected_batches = []
        for batch_start in range(history_size, robot_data.getNTimesteps(), batch_size):
            batch_end = min(batch_start + batch_size, robot_data.getNTimesteps())
            expected_batches.append(os.path.join(bag_path, f"batch_{batch_start}_{batch_end}.h5"))

        batch_files = expected_batches
        all_batches_exist = all(os.path.exists(f) for f in batch_files)

        if not all_batches_exist:
            batch_files = ComputeVicRegData(H, K, plane_normal, plane_distance, robot_data, odom_offset, meters_per_pixel=1/px_meter, history_size=history_size, patch_size=(patch_size, patch_size), batch_size=batch_size, output_dir=bag_path, num_patches=num_patches, lateral_pixel_shift=lateral_pixel_shift)
            print(f"Saved VICReg data batches to {bag_path}")

            missing_files = [f for f in expected_batches if f not in batch_files]
            if missing_files:
                raise RuntimeError(f"Missing batch files: {missing_files}")
            print("All batch files verified successfully")
        else:
            print(f"All batch files already exist: {len(batch_files)} files found")

        gc.collect()
        print("Memory cleared before combining batches")
        H, K, RT, plane_normal, plane_distance, robot_data = None, None, None, None, None, None

        combine_batches_to_single_h5(batch_files, vicreg_data_path)

        if not os.path.exists(vicreg_data_path) or os.path.getsize(vicreg_data_path) == 0:
            raise RuntimeError(f"Failed to save combined HDF5 file: {vicreg_data_path}")
        print(f"Combined .h5 file verified: {vicreg_data_path}")

        gc.collect()
        cleanup_batch_files(batch_files)