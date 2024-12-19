import sys
import time
import torch
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class PointCloudViewer:
    def __init__(self, window_title="3D Point Cloud Viewer", width=800, height=600):
        """
        Initializes the OpenGL context and sets up the viewer.

        :param window_title: Title of the OpenGL window.
        :param width: Width of the window in pixels.
        :param height: Height of the window in pixels.
        """
        # Initialize camera parameters
        self.camera_x, self.camera_y, self.camera_z = 0.0, 0.0, 5.0    # Position
        self.camera_yaw, self.camera_pitch = 0.0, 0.0                 # Orientation
        self.move_speed = 0.1
        self.rotate_speed = 2.0

        # Mouse interaction variables
        self.mouse_left_down = False
        self.mouse_x_prev, self.mouse_y_prev = 0, 0

        # Point cloud data
        self.points = np.empty((0, 3), dtype=np.float32)

        # Flag to control the main loop
        self.running = True

        # Initialize FreeGLUT
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutCreateWindow(window_title.encode('utf-8'))

        # Initialize OpenGL settings
        self.init_gl(width, height)

        # Register callbacks
        glutDisplayFunc(self.display_callback)
        glutReshapeFunc(self.reshape_callback)
        glutKeyboardFunc(self.keyboard_callback)
        glutSpecialFunc(self.special_keys_callback)
        glutMouseFunc(self.mouse_callback)
        glutMotionFunc(self.motion_callback)
        # No need to set an idle function since we control the loop externally

    def init_gl(self, width, height):
        """
        Sets up the OpenGL environment.

        :param width: Width of the window.
        :param height: Height of the window.
        """
        glClearColor(0.0, 0.0, 0.0, 1.0)   # Black background
        glClearDepth(1.0)                  # Clear depth buffer
        glEnable(GL_DEPTH_TEST)            # Enable depth testing
        glDepthFunc(GL_LEQUAL)             # Type of depth test
        glPointSize(3.0)                   # Size of points
        glEnable(GL_POINT_SMOOTH)          # Smooth points

        # Setup projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def set_points(self, points_tensor):
        """
        Sets or updates the point cloud data.

        :param points_tensor: A PyTorch tensor of shape [n_boards, n_pts, 3].
        """
        if not isinstance(points_tensor, torch.Tensor):
            raise TypeError("points_tensor must be a PyTorch tensor.")

        if points_tensor.dim() != 3 or points_tensor.size(2) != 3:
            raise ValueError("points_tensor must have shape [n_boards, n_pts, 3].")

        # For simplicity, take the first board
        self.points = points_tensor[0].cpu().numpy().astype(np.float32)
        glutPostRedisplay()

    def display_callback(self):
        """
        Renders the point cloud.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Apply camera transformations
        glRotatef(-self.camera_pitch, 1.0, 0.0, 0.0)
        glRotatef(-self.camera_yaw, 0.0, 1.0, 0.0)
        glTranslatef(-self.camera_x, -self.camera_y, -self.camera_z)

        # Draw points
        glBegin(GL_POINTS)
        for point in self.points:
            x, y, z = point
            glColor3f((x + 1) / 2, (y + 1) / 2, (z + 1) / 2)  # Color mapping
            glVertex3f(x, y, z)
        glEnd()

        glutSwapBuffers()

    def reshape_callback(self, width, height):
        """
        Adjusts the viewport and projection matrix when the window is resized.

        :param width: New width of the window.
        :param height: New height of the window.
        """
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glutPostRedisplay()

    def keyboard_callback(self, key, x, y):
        """
        Handles standard key presses for camera movement and exiting.

        :param key: The key pressed.
        :param x: X-coordinate of the mouse.
        :param y: Y-coordinate of the mouse.
        """
        key = key.decode('utf-8') if isinstance(key, bytes) else key

        # Calculate forward and right vectors based on yaw
        rad_yaw = np.radians(self.camera_yaw)
        forward_x = np.sin(rad_yaw)
        forward_z = np.cos(rad_yaw)
        right_x = np.cos(rad_yaw)
        right_z = -np.sin(rad_yaw)

        if key == 'w':  # Move forward
            self.camera_x += forward_x * self.move_speed
            self.camera_z += forward_z * self.move_speed
        elif key == 's':  # Move backward
            self.camera_x -= forward_x * self.move_speed
            self.camera_z -= forward_z * self.move_speed
        elif key == 'a':  # Strafe left
            self.camera_x -= right_x * self.move_speed
            self.camera_z -= right_z * self.move_speed
        elif key == 'd':  # Strafe right
            self.camera_x += right_x * self.move_speed
            self.camera_z += right_z * self.move_speed
        elif key == 'q':  # Move down
            self.camera_y -= self.move_speed
        elif key == 'e':  # Move up
            self.camera_y += self.move_speed
        elif key == '\x1b':  # Escape key
            self.exit()

        glutPostRedisplay()

    def special_keys_callback(self, key, x, y):
        """
        Handles special key presses (e.g., arrow keys) for camera rotation.

        :param key: The special key pressed.
        :param x: X-coordinate of the mouse.
        :param y: Y-coordinate of the mouse.
        """
        if key == GLUT_KEY_LEFT:
            self.camera_yaw -= self.rotate_speed
        elif key == GLUT_KEY_RIGHT:
            self.camera_yaw += self.rotate_speed
        elif key == GLUT_KEY_UP:
            self.camera_pitch -= self.rotate_speed
            self.camera_pitch = max(-89.9, self.camera_pitch)  # Prevent flipping
        elif key == GLUT_KEY_DOWN:
            self.camera_pitch += self.rotate_speed
            self.camera_pitch = min(89.9, self.camera_pitch)

        glutPostRedisplay()

    def mouse_callback(self, button, state, x, y):
        """
        Handles mouse button events for initiating camera rotation.

        :param button: The mouse button pressed.
        :param state: The state of the button (pressed/released).
        :param x: X-coordinate of the mouse.
        :param y: Y-coordinate of the mouse.
        """
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.mouse_left_down = True
                self.mouse_x_prev, self.mouse_y_prev = x, y
            else:
                self.mouse_left_down = False

    def motion_callback(self, x, y):
        """
        Handles mouse motion events for rotating the camera.

        :param x: Current X-coordinate of the mouse.
        :param y: Current Y-coordinate of the mouse.
        """
        if self.mouse_left_down:
            dx = x - self.mouse_x_prev
            dy = y - self.mouse_y_prev
            self.mouse_x_prev, self.mouse_y_prev = x, y

            self.camera_yaw += dx * 0.2
            self.camera_pitch += dy * 0.2
            self.camera_pitch = max(-89.9, min(89.9, self.camera_pitch))

            glutPostRedisplay()

    def main_loop_iteration(self):
        """
        Processes a single iteration of the main loop.

        Call this method iteratively to handle events and render frames.
        """
        if not self.running:
            return False  # Indicates that the loop should stop

        try:
            glutMainLoopEvent()
        except RuntimeError:
            # No events to process
            pass
        except Exception as e:
            print(f"Exception in main loop: {e}")
            self.running = False
            return False

        return self.running

    def exit(self):
        """
        Exits the main loop and terminates the application gracefully.
        """
        self.running = False
        glutLeaveMainLoop()

# Example Usage
if __name__ == "__main__":
    import threading

    # Initialize the viewer
    viewer = PointCloudViewer()

    # Create initial random points
    n_boards = 1
    n_pts = 1000
    initial_points = torch.rand((n_boards, n_pts, 3)) * 2.0 - 1.0  # Points in [-1,1]^3
    viewer.set_points(initial_points)

    # Define a function to run the main loop iteratively
    def run_viewer():
        while viewer.main_loop_iteration():
            # Here you can add additional processing if needed
            # For example, updating points dynamically
            time.sleep(0.016)  # Approximately 60 FPS

    # Run the viewer in the main thread
    try:
        run_viewer()
    except KeyboardInterrupt:
        viewer.exit()
        sys.exit(0)
