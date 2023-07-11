from collections import namedtuple
import time
import os
import numpy as np
from WindowRender import init, render, window_active, end
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local
from pyrr import Matrix44, matrix44, Vector3
from Physics import Body, Force, Velocity, apply_rot_force, apply_trans_force, earth_g_force, lin_air_drag, rot_air_torque


SpatialObject = namedtuple('SpatialObject', 'body coordinateSystem vel')

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

def rotate_force_to_local(force, coordinate_system):
    return Force(
        rotate_to_local(coordinate_system, force.dir),
        force.pos,
        force.magnitude)

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
        magnitude=3.05)
    f2 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([0.5, 0.0, 0.5]),
        magnitude=3.0)
    f3 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([0.5, 0.0, -0.5]),
        magnitude=3.0)
    f4 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([-0.5, 0.0, -0.5]),
        magnitude=3.05)
    start = time.time()
    time_passed = 0
    prev_frame = 0

    window, shader, VAO = init(vertices, indices)

    # while window_active(window):
    while time_passed < 10.0:
        now = time.time()
        delta_time = now - (prev_frame if prev_frame != 0 else now)
        prev_frame = now

        # os.system('cls')
        # print(spatialObject)
        # print(delta_time)
        #time.sleep(0.002)

        rot_air_torque_ = rot_air_torque(spatialObject.vel.rot)

        rot_axis, rot_angle, rot_vel_ = apply_rot_force(
            [f1, f2, f3, f4],
            spatialObject.vel.rot,
            delta_time,
            spatialObject.body,
            rot_air_torque_)
        coordinate_system_ = rotate(
            spatialObject.coordinateSystem,
            rot_axis,
            rot_angle)

        g = rotate_force_to_local(
            earth_g_force(spatialObject.body.mass),
            coordinate_system_)

        lin_air_drag_ = rotate_force_to_local(
            lin_air_drag(spatialObject.vel.lin),
            coordinate_system_)

        origin_delta, lin_vel_ = apply_trans_force(
            [f1, f2, f3, f4, g, lin_air_drag_],
            rotate_to_local(coordinate_system_, spatialObject.vel.lin),
            delta_time,
            spatialObject.body)
        coordinate_system_ = translate(
            coordinate_system_,
            rotate_to_global(coordinate_system_, origin_delta))
        
        spatialObject = SpatialObject(
            spatialObject.body,
            coordinate_system_,
            Velocity(
                rotate_to_global(coordinate_system_, lin_vel_),
                rot_vel_))

        render(
            window, 
            shader, 
            VAO, 
            indices, 
            0,
            -60,
            Matrix44.from_matrix33(coordinate_system_.rotation),
            Matrix44.from_translation(Vector3(coordinate_system_.origin)))

        frame_count += 1
        time_passed = now - start
    print(spatialObject)
    print("time_passed: " + str(time_passed))
    print("fps: " + str(frame_count / 10.0))
    print("air_drag: " + str(lin_air_drag_))

run()
end()
