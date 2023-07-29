### TODO: ###
# Simple Manual Sim (1m)
# X- keyboard input controlling forces
# X- refactor Sim Module
# X- engine module with side forces
# - complex shapes
# - HUD with metrics
# - drone type (maybe module as well)
# - improved lighting with better normals, different colors for each side
# (PID-controller, Unit tests, xbox controller input, (controll assistant: dissaster recovery, auto hover), winds)
#_____________________________
# CFD calculated air drags (1-2m)
#_____________________________
# AI learning to fly sim (1-3m)
#_____________________________
# use AI model in DIY mini drone using IoT  (1-2m)
#_____________________________
# drone fleet interaction (2-4m)
#_____________________________



from collections import namedtuple
import time
import os
import numpy as np
from WindowRender import init, render, window_active, end
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local
from pyrr import Matrix44, matrix44, Vector3
from Physics import Body, Force, Velocity, apply_rot_force, apply_trans_force, earth_g_force, lin_air_drag, rot_air_torque
from Geometry import Cube, create_cube, area
from KeyboardControler import poll_keyboard
from Sim import init_sim, step_sim, SpatialObject


def stop():
    return [
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([-0.5, 0.0, 0.5]),
            magnitude=0.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, 0.5]),
            magnitude=0.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, -0.5]),
            magnitude=0.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([-0.5, 0.0, -0.5]),
            magnitude=0.0)]

def yaw():
    return [
        Force(
            dir=np.array([0.0, 0.0, 1.0]),
            pos=np.array([-0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, -0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, 1.0]),
            pos=np.array([-0.5, -0.5, 0.0]),
            magnitude=3.0)]

def forward():
    return [
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([-0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, -0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([-0.5, -0.5, 0.]),
            magnitude=3.0)]

def pitch():
    return [
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([-0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, 1.0]),
            pos=np.array([0.5, -0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, 1.0]),
            pos=np.array([-0.5, -0.5, 0.]),
            magnitude=3.0)]

def roll():
    return [
        Force(
            dir=np.array([0.0, -1.0, 0.0]),
            pos=np.array([-0.5, 0.0, -0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, -0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, 0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, -1.0, 0.0]),
            pos=np.array([-0.5, 0.0, 0.5]),
            magnitude=3.0)]

def run():
    (spatial_object, frame_count, prev_frame, engine_spec) = init_sim()
    vertices = np.array([
        *spatial_object.body.cube.left_bottom_inner_corner, 0.0, 0.0, -1.0,
        *spatial_object.body.cube.right_bottom_inner_corner, 0.0, 0.0, -1.0,
        *spatial_object.body.cube.right_top_inner_corner, 0.0, 0.0, -1.0,
        *spatial_object.body.cube.left_top_inner_corner, 0.0, 0.0, -1.0,
        *spatial_object.body.cube.left_bottom_outer_corner, 0.0, 0.0, -1.0,
        *spatial_object.body.cube.right_bottom_outer_corner, 0.0, 0.0, -1.0,
        *spatial_object.body.cube.right_top_outer_corner, 0.0, 0.0, -1.0,
        *spatial_object.body.cube.left_top_outer_corner, 0.0, 0.0, -1.0
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 2, 3, 0,    # Front face
        1, 5, 6, 6, 2, 1,    # Right face
        7, 6, 5, 5, 4, 7,    # Back face
        4, 0, 3, 3, 7, 4,    # Left face
        4, 5, 1, 1, 0, 4,    # Bottom face
        3, 2, 6, 6, 7, 3     # Top face
    ], dtype=np.uint32)

    forces = [
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([-0.5, 0.0, 0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, 0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, -0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([-0.5, 0.0, -0.5]),
            magnitude=3.0)]
    start = time.time()
    time_passed = 0
    prev_frame = 0

    window, shader, VAO = init(vertices, indices)

    while window_active(window):
        imput = poll_keyboard()
        (spatial_object,
            frame_count,
            prev_frame
        ) = step_sim(
            spatial_object,
            frame_count,
            prev_frame,
            imput,
            engine_spec)

        render(
            window, 
            shader, 
            VAO, 
            indices, 
            0,
            -20,
            Matrix44.from_matrix33(
                spatial_object.coordinateSystem.rotation),
            Matrix44.from_translation(
                Vector3(spatial_object.coordinateSystem.origin)))

    print(spatial_object)
    print("frame_count: " + str(frame_count))

run()
end()
