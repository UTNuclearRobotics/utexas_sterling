import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
from simulate_driving_image import *

# Homography matrix (example)
# homography = np.array(
#     [[-1.5563e+02,  0.0000e+00,  5.3390e+05],
#         [-1.1038e+02,  1.0000e+01,  3.5589e+05],
#         [-8.7156e-02,  0.0000e+00,  2.8100e+02]
# ], dtype=np.float32)

# homography = np.array([
#     [1.0, 0.2, -0.5],
#     [0.1, 1.0, -0.3],
#     [0.001, 0.002, 1.0]
# ], dtype=np.float32)

homography = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Vertex Shader (GLSL)
vertex_shader = """
#version 330 core
layout(location = 0) in vec2 aPos;   // Input vertex positions (clip space)
out vec2 TexCoords;                 // Pass to fragment shader

uniform mat3 uHomography;           // Homography matrix

void main()
{
    // Apply the homography transformation
    vec3 pos = uHomography * vec3(aPos, 1.0);
    
    // Clip pixels behind the camera (w <= 0)
    if (pos.z <= 0.0)
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0); // Move it out of view
    else
        gl_Position = vec4(pos.xy / pos.z, 0.0, 1.0); // Perspective division

    // Map vertex positions to texture coordinates ([-1, 1] -> [0, 1])
    TexCoords = aPos * 0.5 + 0.5;
}
"""

# Fragment Shader (GLSL)
fragment_shader = """
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D uTexture;

void main()
{
    // Sample the texture
    FragColor = texture(uTexture, TexCoords);
}
"""

# Initialize GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

# Create a windowed mode window and OpenGL context
window = glfw.create_window(800, 600, "Homography Transformation", None, None)
glfw.make_context_current(window)

# Compile and link shaders
shader_program = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

# Set up vertex data (full-screen quad)
vertices = np.array([
    -1.0, -1.0,  # Bottom-left
     1.0, -1.0,  # Bottom-right
    -1.0,  1.0,  # Top-left
     1.0,  1.0   # Top-right
], dtype=np.float32)

# Load vertex data into GPU
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)

glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(0)

# Load texture using Pillow
image = Image.open("/home/justin/MitchLab/utexas_sterling/datasets/big_duck.jpg").transpose(Image.FLIP_TOP_BOTTOM)
img_data = np.array(image, dtype=np.uint8)

texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

print(homography)

# Pass homography matrix to shader
glUseProgram(shader_program)
uHomographyLoc = glGetUniformLocation(shader_program, "uHomography")
glUniformMatrix3fv(uHomographyLoc, 1, GL_FALSE, homography)

# T1 = torch.tensor([
#         [1, 0, -image.width / 2],
#         [0, 1, -image.height / 2],
#         [0, 0, 1]
#     ])

# # Translation to shift back after the homography
# T2 = torch.tensor([
#     [1, 0, image.width / 2],
#     [0, 1, image.height / 2],
#     [0, 0, 1]
# ])

# Main render loop
angle_incr = math.radians(1)
f = 10
while not glfw.window_should_close(window):
    H = camera_intrinsic_matrix(f, 0, 0) @ normalized_homography_vect_vect(R, T)
    # H = T1 @ H @ T2
    homography = H.numpy()
    glUniformMatrix3fv(uHomographyLoc, 1, GL_FALSE, homography)

    glClear(GL_COLOR_BUFFER_BIT)

    # Draw the full-screen quad
    glBindTexture(GL_TEXTURE_2D, texture)
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    glfw.swap_buffers(window)
    glfw.poll_events()

    if glfw.get_key(window, glfw.KEY_I) == glfw.PRESS:
        R.__setitem__((0, 0), R[0, 0] - angle_incr)
    if glfw.get_key(window, glfw.KEY_K) == glfw.PRESS:
        R.__setitem__((0, 0), R[0, 0] + angle_incr)
    if glfw.get_key(window, glfw.KEY_J) == glfw.PRESS:
        R.__setitem__((1, 0), R[1, 0] - angle_incr)
    if glfw.get_key(window, glfw.KEY_L) == glfw.PRESS:
        R.__setitem__((1, 0), R[1, 0] + angle_incr)
    if glfw.get_key(window, glfw.KEY_O) == glfw.PRESS:
        R.__setitem__((2, 0), R[2, 0] - angle_incr)
    if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS:
        R.__setitem__((2, 0), R[2, 0] + angle_incr)


    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        T.__setitem__((0, 0), T[0, 0] - 0.1)
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        T.__setitem__((0, 0), T[0, 0] + 0.1)

    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        T.__setitem__((1, 0), T[1, 0] + 0.1)
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        T.__setitem__((1, 0), T[1, 0] - 0.1)

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        T.__setitem__((2, 0), T[2, 0] + 0.1)
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        T.__setitem__((2, 0), T[2, 0] - 0.1)

    if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS:
        f *= 1.1
    if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
        f /= 1.1

    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    print("H:   ", H)

# Clean up
glDeleteVertexArrays(1, [VAO])
glDeleteBuffers(1, [VBO])
glDeleteTextures(1, [texture])
glfw.terminate()
