from WindowRender import init, render, window_active, end
import numpy as np
from pyrr import Matrix44, matrix44
import time


# Initialize FPS counter variables
frame_count = 0
start_time = time.time()

vertices = np.array([
    -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
     0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
     0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
    -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
    -0.5, -0.5,  0.5,  0.0,  0.0,  -1.0,
     0.5, -0.5,  0.5,  0.0,  0.0,  -1.0,
     0.5,  0.5,  0.5,  0.0,  0.0,  -1.0,
    -0.5,  0.5,  0.5,  0.0,  0.0,  -1.0
], dtype=np.float32)

indices = np.array([
    0, 1, 2, 2, 3, 0,    # Front face
    1, 5, 6, 6, 2, 1,    # Right face
    7, 6, 5, 5, 4, 7,    # Back face
    4, 0, 3, 3, 7, 4,    # Left face
    4, 5, 1, 1, 0, 4,    # Bottom face
    3, 2, 6, 6, 7, 3     # Top face
], dtype=np.uint32)

window, shader, VAO = init(vertices, indices)

# Main loop
rotation_angle = 0.0
while window_active(window):
    rotation_x = Matrix44.from_x_rotation(np.radians(rotation_angle))  # Rotate by rotation_angle degrees around x-axis
    rotation_y = Matrix44.from_y_rotation(np.radians(rotation_angle))  # Rotate by rotation_angle degrees around y-axis
    rotation = matrix44.multiply(rotation_x, rotation_y)  # Combine the two rotations
    render(window, shader, VAO, indices, -10, rotation)
    rotation_angle += 0.075
    if rotation_angle >= 360.0:  # Keep rotation_angle between 0 and 359
        rotation_angle -= 360.0

    # FPS counting
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time >= 1:  # Every second
        print(f"FPS: {frame_count / elapsed_time}")
        frame_count = 0
        start_time = current_time        

end()
