import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
import glm  # Install via `pip install PyGLM`

# Vertex Shader
vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoords;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

void main()
{
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
    TexCoords = aTexCoord;
}
"""

# Fragment Shader
fragment_shader = """
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D uTexture;

void main()
{
    FragColor = texture(uTexture, TexCoords);
}
"""

# Initialize GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

# Create a window
window = glfw.create_window(800, 600, "Single Texture Floor", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")
glfw.make_context_current(window)

# Compile shaders
shader_program = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

# Define a large quad as the floor with adjusted texture coordinates
floor_vertices = np.array([
    # Positions            # Texture Coords
    -50.0, 0.0, -50.0,     0.0, 0.0,  # Bottom-left
     50.0, 0.0, -50.0,     1.0, 0.0,  # Bottom-right
    -50.0, 0.0,  50.0,     0.0, 1.0,  # Top-left
     50.0, 0.0,  50.0,     1.0, 1.0,  # Top-right
], dtype=np.float32)

indices = np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32)

# Generate buffers
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)
EBO = glGenBuffers(1)

glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, floor_vertices.nbytes, floor_vertices, GL_STATIC_DRAW)

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))  # Positions
glEnableVertexAttribArray(0)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))  # Texture Coords
glEnableVertexAttribArray(1)

# Load texture
image = Image.open("/home/justin/MitchLab/utexas_sterling/datasets/big_duck.jpg").transpose(Image.FLIP_TOP_BOTTOM)
img_data = np.array(image, dtype=np.uint8)
texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# Camera setup
camera_pos = glm.vec3(0.0, 5.0, 10.0)
camera_front = glm.vec3(0.0, 0.0, -1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0)
camera_speed = 0.5

# Handle WASD movement
def key_callback(window, key, scancode, action, mods):
    global camera_pos, camera_front

    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_W:
            camera_pos += camera_speed * camera_front
        if key == glfw.KEY_S:
            camera_pos -= camera_speed * camera_front
        if key == glfw.KEY_A:
            camera_pos -= glm.normalize(glm.cross(camera_front, camera_up)) * camera_speed
        if key == glfw.KEY_D:
            camera_pos += glm.normalize(glm.cross(camera_front, camera_up)) * camera_speed

glfw.set_key_callback(window, key_callback)

# Set up transformation matrices
projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)
model = glm.mat4(1.0)

glUseProgram(shader_program)
model_loc = glGetUniformLocation(shader_program, "uModel")
view_loc = glGetUniformLocation(shader_program, "uView")
proj_loc = glGetUniformLocation(shader_program, "uProjection")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))
glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))

# Main loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    # Update view matrix based on camera position
    view = glm.lookAt(camera_pos, camera_pos + camera_front, camera_up)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))

    # Render
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    glBindTexture(GL_TEXTURE_2D, texture)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

# Cleanup
glfw.terminate()
