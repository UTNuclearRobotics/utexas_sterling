import sys
import torch
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Example: Suppose you have a tensor [n_boards, n_pts, 3].
n_boards = 1
n_pts = 1000
points_tensor = torch.rand((n_boards, n_pts, 3)) * 2.0 - 1.0  # random points in [-1,1]^3
points = points_tensor[0].cpu().numpy()  # shape (n_pts, 3)

# Camera parameters
camera_x, camera_y, camera_z = 0.0, 0.0, 3.0    # Start at (0,0,3)
camera_yaw, camera_pitch = 0.0, 0.0            # Rotation around Y (yaw) and X (pitch)
move_speed = 0.1
rotate_speed = 2.0

# Mouse interaction
mouse_left_down = False
mouse_x, mouse_y = 0, 0

def init_gl(width, height):
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glPointSize(3.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Apply camera rotations and translations:
    # The order of transformations is important:
    # 1. Rotate by pitch around X, then yaw around Y
    # 2. Translate the camera
    glRotatef(-camera_pitch, 1.0, 0.0, 0.0)
    glRotatef(-camera_yaw, 0.0, 1.0, 0.0)
    glTranslatef(-camera_x, -camera_y, -camera_z)

    # Draw points
    glBegin(GL_POINTS)
    for x, y, z in points:
        glColor3f((x+1)/2, (y+1)/2, (z+1)/2)
        glVertex3f(x, y, z)
    glEnd()

    glutSwapBuffers()

def reshape(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    global camera_x, camera_y, camera_z
    global camera_yaw, camera_pitch

    # Convert camera_yaw to radians for direction calculations
    rad_yaw = np.radians(camera_yaw)

    # Forward vector in the XZ-plane
    forward_x = np.sin(rad_yaw)
    forward_z = np.cos(rad_yaw)
    # Right vector (perpendicular in XZ)
    right_x = np.cos(rad_yaw)
    right_z = -np.sin(rad_yaw)

    if key == b'w':  # Move forward
        camera_x += forward_x * move_speed
        camera_z += forward_z * move_speed
    elif key == b's':  # Move backward
        camera_x -= forward_x * move_speed
        camera_z -= forward_z * move_speed
    elif key == b'a':  # Strafe left
        camera_x -= right_x * move_speed
        camera_z -= right_z * move_speed
    elif key == b'd':  # Strafe right
        camera_x += right_x * move_speed
        camera_z += right_z * move_speed
    elif key == b'q':  # Move down
        camera_y -= move_speed
    elif key == b'e':  # Move up
        camera_y += move_speed
    elif key == b'\x1b':  # Escape key
        glutLeaveMainLoop()  # Exit the main loop

    glutPostRedisplay()

def special_keys(key, x, y):
    global camera_yaw, camera_pitch

    # Rotate the camera with arrow keys
    if key == GLUT_KEY_LEFT:
        camera_yaw -= rotate_speed
    elif key == GLUT_KEY_RIGHT:
        camera_yaw += rotate_speed
    elif key == GLUT_KEY_UP:
        camera_pitch -= rotate_speed
        # Limit pitch to avoid flipping
        camera_pitch = max(-89.9, camera_pitch)
    elif key == GLUT_KEY_DOWN:
        camera_pitch += rotate_speed
        camera_pitch = min(89.9, camera_pitch)

    glutPostRedisplay()

def mouse(button, state, x, y):
    global mouse_left_down, mouse_x, mouse_y
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            mouse_left_down = True
            mouse_x, mouse_y = x, y
        else:
            mouse_left_down = False

def motion(x, y):
    global mouse_x, mouse_y, camera_yaw, camera_pitch
    if mouse_left_down:
        dx = x - mouse_x
        dy = y - mouse_y
        mouse_x, mouse_y = x, y

        # Adjust camera angles based on mouse movement
        camera_yaw += dx * 0.2
        camera_pitch += dy * 0.2
        camera_pitch = max(-89.9, min(89.9, camera_pitch))
        glutPostRedisplay()

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"3D Point Cloud Viewer")
    init_gl(800, 600)
    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special_keys)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutMainLoop()

if __name__ == "__main__":
    main()
