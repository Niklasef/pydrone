from collections import namedtuple
import time
import os
import numpy as np
from WindowRender import init, render, window_active, end
from pyrr import Matrix44, matrix44, Vector3


Body = namedtuple('Body', 'mass')
Velocity = namedtuple('Velocity', 'lin rot')
SpatialObject = namedtuple('SpatialObject', 'body pos vel')
Force = namedtuple('Force', 'dir magnitude')


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


def run():
    frame_count = 0
    body = Body(1.0)
    vel = Velocity(lin=np.zeros(3), rot=np.zeros(3))
    spatialObject = SpatialObject(body=body, pos=np.zeros(3), vel=vel)
    f1 = Force(np.array([0.0, 1.0, 0.0]), 2.0)
    start = time.time()
    time_passed = 0
    prev_frame = 0


    window, shader, VAO = init(vertices, indices)
    translation = Matrix44.from_translation(Vector3([0.0, 0.0, 0.0]))
    rotation_angle = 0.0
    rotation_x = Matrix44.from_x_rotation(np.radians(rotation_angle))  # Rotate by rotation_angle degrees around x-axis
    rotation_y = Matrix44.from_y_rotation(np.radians(rotation_angle))  # Rotate by rotation_angle degrees around y-axis
    rotation = matrix44.multiply(rotation_x, rotation_y)  # Combine the two rotations    

    while window_active(window):
        now = time.time()
        delta_time = now - (prev_frame if prev_frame != 0 else now)
        prev_frame = now

        # os.system('cls')
        # print(spatialObject)
        # print(delta_time)
        #time.sleep(0.002)
        lin_acc = (f1.magnitude * f1.dir) / spatialObject.body.mass
        lin_vel_ = spatialObject.vel.lin + (lin_acc * delta_time)
        vel_ = Velocity(lin=lin_vel_, rot=spatialObject.vel.rot)
        pos_ = spatialObject.pos + (lin_vel_ * delta_time)
        spatialObject = SpatialObject(body, pos_, vel_)

        translation = Matrix44.from_translation(Vector3(spatialObject.pos))

        render(window, shader, VAO, indices, -15, -50, rotation, translation)

        frame_count += 1
        time_passed = now - start
    print(spatialObject)
    print("time_passed: " + str(time_passed))
    print("fps: " + str(frame_count / 10.0))

run()
end()
