from collections import namedtuple
import time
import os
import numpy as np
from WindowRender import init, render, window_active, end
from CoordinateSystem import CoordinateSystem, rotate, translate
from pyrr import Matrix44, matrix44, Vector3


Body = namedtuple('Body', 'mass')
Velocity = namedtuple('Velocity', 'lin rot')
SpatialObject = namedtuple('SpatialObject', 'body coordinateSystem vel')
Force = namedtuple('Force', 'dir pos magnitude')

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
    body = Body(mass=1.0)
    coordinateSystem = CoordinateSystem(
        origin=np.zeros(3),
        rotation=np.eye(3))
    vel = Velocity(lin=np.zeros(3), rot=np.zeros(3))
    spatialObject = SpatialObject(
        body,
        coordinateSystem,
        vel)
    f1 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([-0.5, 0.0, 0.5]),
        magnitude=1)
    f1_start = 1
    f1_end = 10
    start = time.time()
    time_passed = 0
    prev_frame = 0

    window, shader, VAO = init(vertices, indices)

    inertia = spatialObject.body.mass * (1.0 * 1.0) * (3.0/10.0) #  when force at corner and acting perpendicular to face
    # while window_active(window):
    while time_passed < 10.0:
        now = time.time()
        delta_time = now - (prev_frame if prev_frame != 0 else now)
        prev_frame = now

        # os.system('cls')
        # print(spatialObject)
        # print(delta_time)
        #time.sleep(0.002)

        if time_passed > f1_start and time_passed < f1_end:
            f1_ = Force(dir=f1.dir, pos=f1.pos, magnitude=f1.magnitude)
        else:
            f1_ = Force(dir=f1.dir, pos=f1.pos, magnitude=0.0)

        tourque = np.cross(f1_.pos, f1_.dir * f1_.magnitude)
        rot_acc = tourque / inertia
        rot_vel_delta = rot_acc * delta_time
        rot_vel_ = spatialObject.vel.rot + rot_vel_delta
        rot_speed = np.linalg.norm(rot_vel_)
        rot_axis = rot_vel_ / rot_speed if rot_speed > 0 else np.array([1.0, 0.0, 0.0])
        rot_angle = rot_speed * delta_time

        coordinate_system_ = rotate(spatialObject.coordinateSystem, rot_axis, rot_angle)

        lin_acc = (f1_.magnitude * f1_.dir) / spatialObject.body.mass
        lin_vel_delta = lin_acc * delta_time
        lin_vel_delta_rotated = matrix44.apply_to_vector(
            Matrix44.from_matrix33(coordinate_system_.rotation),
            lin_vel_delta)        
        lin_vel_ = spatialObject.vel.lin + lin_vel_delta_rotated
        origin_delta = lin_vel_ * delta_time

        coordinate_system_ = translate(coordinate_system_, origin_delta)
        spatialObject = SpatialObject(
            spatialObject.body,
            coordinate_system_,
            Velocity(lin_vel_, rot_vel_))

        render(
            window, 
            shader, 
            VAO, 
            indices, 
            0, 
            -15, 
            Matrix44.from_matrix33(coordinate_system_.rotation),
            Matrix44.from_translation(Vector3(coordinate_system_.origin)))

        frame_count += 1
        time_passed = now - start
    print(spatialObject)
    print("time_passed: " + str(time_passed))
    print("fps: " + str(frame_count / 10.0))

run()
end()
