import cv2
from homography import *
import math
import os
import tkinter as tk
import torch
from utils import *

def rotation_matrix_x(theta):
    return torch.tensor([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ], dtype=torch.float32)

def rotation_matrix_y(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return torch.tensor([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ], dtype=torch.float32)

def rotation_matrix_z(theta):
    return torch.tensor([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

def rotate_xyz(rotation_vector):
    return rotation_matrix_x(rotation_vector[0,0]) @ rotation_matrix_y(rotation_vector[1,0]) @ rotation_matrix_z(rotation_vector[2,0])

def camera_intrinsic_matrix(f, u0, v0):
    return torch.tensor([
        [f, 0, u0],
        [0, f, v0],
        [0, 0, 1]
    ], dtype=torch.float32)

def normalized_homography_mat_vect(rotation_matrix, translation_vector):
    return torch.cat((rotation_matrix[:, :2], translation_vector), dim=1)

def normalized_homography_vect_vect(rotation_vector, translation_vector):
    return torch.cat((rotate_xyz(rotation_vector)[:, :2], translation_vector), dim=1)

run = True
show_depth = False

T = torch.tensor([[0.0], [0.0], [1.0]])
R = torch.tensor([[0.0], [0.0], [0.0]])

def quit():
    global run
    run = False
    
def reset_transform():
    global T
    global R
    T = torch.tensor([[0.0], [0.0], [1.0]])
    R = torch.tensor([[0.0], [0.0], [0.0]])

def toggle_depth():
    global show_depth
    show_depth = not show_depth


if __name__ == "__main__":
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()


    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    image_dir = script_dir + "/../datasets/"
    path_to_image = image_dir + "big_duck.jpg"

    map_image = cv2.imread(path_to_image)
    image_height, image_width, channels = map_image.shape
    # print("H:   ", height, "    W:  ", width)
    
    scale_width = screen_width / image_width
    scale_height = screen_height / image_height
    scale = min(scale_width, scale_height, 1)  # Ensure we don't upscale

    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    #105-108 ijkl
    angle_incr = math.radians(1)
    actions = {
        113: quit,                                                      #q          Quit
        101: reset_transform,                                           #e          Reset Transform
        92: toggle_depth,                                               #\|         Toggle Depth
        97: lambda: T.__setitem__((0, 0), T[0, 0] - 10),                #a          Translate X
        100: lambda: T.__setitem__((0, 0), T[0, 0] + 10),               #d
        119: lambda: T.__setitem__((1, 0), T[1, 0] - 10),               #w          Translate Y
        115: lambda: T.__setitem__((1, 0), T[1, 0] + 10),               #s
        82: lambda: T.__setitem__((2, 0), T[2, 0] + 10),              #up arrow   Translate Z
        84: lambda: T.__setitem__((2, 0), T[2, 0] - 10),              #down arrow

        105: lambda: R.__setitem__((0, 0), R[0, 0] - angle_incr),       #i          Rotate X
        107: lambda: R.__setitem__((0, 0), R[0, 0] + angle_incr),       #k
        106: lambda: R.__setitem__((1, 0), R[1, 0] - angle_incr),       #j          Rotate Y
        108: lambda: R.__setitem__((1, 0), R[1, 0] + angle_incr),       #l
        'c': lambda: print("You pressed C!"),
    }

    cv2.namedWindow("Fullscreen Image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Fullscreen Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while run:
        H = camera_intrinsic_matrix(10, image_width / 2, image_height / 2) @ \
            normalized_homography_vect_vect(R, T)
        # H = normalized_homography_vect_vect(R, T)
        print(H)
        H = H.numpy()
        image_to_show = fixedWarpPerspective(H, map_image) if show_depth else \
            cv2.warpPerspective(map_image, H, (image_width * 2, image_height * 2), \
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # map_image_rgba = cv2.cvtColor(map_image, cv2.COLOR_BGR2BGRA)

        # resized_image = cv2.resize(image_to_show, (new_width, new_height))
        cv2.imshow("Fullscreen Image", image_to_show)
        key = cv2.waitKey(0)
        print("key: ", key)
        if key in actions:
            actions[key]()

    
    cv2.destroyAllWindows()