from model_chessboard import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
from scipy.spatial.transform import *


def rotation_x(theta):
    """4x4 rotation matrix about the x-axis by angle theta (radians)."""
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta),-np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(theta):
    """3x3 rotation matrix about the y-axis by angle theta (radians)."""
    return np.array([
        [ np.cos(theta),  0, np.sin(theta), 0],
        [             0,  1,             0, 0],
        [-np.sin(theta),  0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def translation_z(distance):
    """4x4 rotation matrix about the x-axis by angle theta (radians)."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, distance],
        [0, 0, 0, 1]
    ])

class ChessboardRenderer:
    def __init__(self):
        self.camera_rt = np.array(
            [
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, 1]
            ])
        self.camera_speed = 5.0  # Movement speed
        self.dt = 1.0 / 60.0  # Default delta time (60 FPS)
        self.mouselook_active = False
        self.mouse_sensitivity = 0.01  # Adjust for finer control
        self.kb_lr = 0.5
        self.model_cb = ModelChessboard()

        pygame.init()
        self.display = (1000, 1000)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        # screen = pygame.display.set_mode((0, 0), DOUBLEBUF | OPENGL | FULLSCREEN)
        glEnable(GL_DEPTH_TEST)
        glPointSize(5.0)  # Set point size
        glClearColor(0.0, 0.0, 0.0, 1.0)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        self.init_projection()
        self.clock = pygame.time.Clock()
        self.running = True   

    def init_projection(self):
        """Initializes the OpenGL scene with a projection matrix."""
        glMatrixMode(GL_PROJECTION)  # Set the projection matrix
        glLoadIdentity()
        gluPerspective(60, 1, 0.01, 1000.0)  # 60° FOV, aspect ratio, near and far planes

    def update_and_draw(self, rt_list):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)  # Ensure we're modifying the modelview matrix
        glLoadIdentity()  # Reset the modelview matrix
        glMultMatrixf(self.camera_rt.T.flatten())  # Apply the rigid transformation

        glColor3f(1, 0, 1)  # Set the line color (white)
        self.model_cb.draw(rt_list)
        # self.model_cb.drawLines()
        # draw_chessboard()  # Draw the chessboard
        pygame.display.flip()

    def handle_mouse_motion(self):
        """Handles mouse movements for rotating the camera like in a first-person shooter."""

    def move_camera(self, keys, speed=0.1):
        if keys[pygame.K_a]:  # Left
            Ry = rotation_y(-0.1)
            self.camera_rt = Ry @ self.camera_rt
        if keys[pygame.K_d]:  # Right
            Ry = rotation_y(0.1)
            self.camera_rt = Ry @ self.camera_rt
        if keys[pygame.K_w]:  # Forward
            Tz = translation_z(0.1)
            self.camera_rt = Tz @ self.camera_rt
        if keys[pygame.K_s]:  # Back
            Tz = translation_z(-0.1)
            self.camera_rt = Tz @ self.camera_rt
        if keys[pygame.K_q]:  # Quit
            self.running = False   

        # if mouselook_active:
        #     x, y = pygame.mouse.get_rel()  # Get relative mouse motion
        #     Rx = rotation_x(-x * mouse_sensitivity)
        #     Ry = rotation_y(0 * mouse_sensitivity)
        #     camera_rt = Rx @ Ry @ camera_rt

    def display_iteration(self, rt_list):
        self.dt = self.clock.tick(60) / 1000.0  # Calculate delta time in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button pressed
                mouselook_active = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # Left mouse button released
                mouselook_active = False

        keys = pygame.key.get_pressed()  # Get all key states
        self.move_camera(keys, speed=5.0)
        self.handle_mouse_motion()

        self.update_and_draw(rt_list)
        self.clock.tick(60)

if __name__ == "__main__":
    cbr = ChessboardRenderer()

    # while cbr.running:
    #     cbr.display_iteration()

    # pygame.quit()