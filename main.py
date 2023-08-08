### TODO: ###
# Simple Manual Sim (1m)
# X- keyboard input controlling forces
# X- refactor Sim Module
# X- engine module with side forces
# X- rectangle shape
# X- drone module
# X- multi body drone
# X- rotatable multi body shapes
# - engines distrubuted across multiple shapes
# - HUD with metrics
# - improved lighting with better normals, different colors for each side
# (PID-controller, Unit tests, xbox controller input, (controll assistant: dissaster recovery, auto hover), winds, complex detailed shapes, refactor force type to be single vector not split in magnitude - or possible easy to convert between these two forms? maybe force module?)
# - self-level mode, angle mode, horizon mode, acro mode 
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
from Cube import Cube, create_cube, area
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
    (frame_count, prev_frame, drone) = init_sim()

    vertices = np.array([
        *rotate_to_global(drone.spatial_objects[0].coordinateSystem, drone.spatial_objects[0].body.shape.left_bottom_inner_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[0].coordinateSystem, drone.spatial_objects[0].body.shape.right_bottom_inner_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[0].coordinateSystem, drone.spatial_objects[0].body.shape.right_top_inner_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[0].coordinateSystem, drone.spatial_objects[0].body.shape.left_top_inner_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[0].coordinateSystem, drone.spatial_objects[0].body.shape.left_bottom_outer_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[0].coordinateSystem, drone.spatial_objects[0].body.shape.right_bottom_outer_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[0].coordinateSystem, drone.spatial_objects[0].body.shape.right_top_outer_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[0].coordinateSystem, drone.spatial_objects[0].body.shape.left_top_outer_corner), 0.0, 0.0, -1.0,

        *rotate_to_global(drone.spatial_objects[1].coordinateSystem, drone.spatial_objects[1].body.shape.left_bottom_inner_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[1].coordinateSystem, drone.spatial_objects[1].body.shape.right_bottom_inner_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[1].coordinateSystem, drone.spatial_objects[1].body.shape.right_top_inner_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[1].coordinateSystem, drone.spatial_objects[1].body.shape.left_top_inner_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[1].coordinateSystem, drone.spatial_objects[1].body.shape.left_bottom_outer_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[1].coordinateSystem, drone.spatial_objects[1].body.shape.right_bottom_outer_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[1].coordinateSystem, drone.spatial_objects[1].body.shape.right_top_outer_corner), 0.0, 0.0, -1.0,
        *rotate_to_global(drone.spatial_objects[1].coordinateSystem, drone.spatial_objects[1].body.shape.left_top_outer_corner), 0.0, 0.0, -1.0,
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 2, 3, 0,    # Front face
        1, 5, 6, 6, 2, 1,    # Right face
        7, 6, 5, 5, 4, 7,    # Back face
        4, 0, 3, 3, 7, 4,    # Left face
        4, 5, 1, 1, 0, 4,    # Bottom face
        3, 2, 6, 6, 7, 3,     # Top face

        0+8, 1+8, 2+8, 2+8, 3+8, 0+8,    # Front face
        1+8, 5+8, 6+8, 6+8, 2+8, 1+8,    # Right face
        7+8, 6+8, 5+8, 5+8, 4+8, 7+8,    # Back face
        4+8, 0+8, 3+8, 3+8, 7+8, 4+8,    # Left face
        4+8, 5+8, 1+8, 1+8, 0+8, 4+8,    # Bottom face
        3+8, 2+8, 6+8, 6+8, 7+8, 3+8     # Top face
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
        (frame_count,
            prev_frame,
            drone
        ) = step_sim(
            frame_count,
            prev_frame,
            imput,
            drone)

        render(
            window, 
            shader, 
            VAO, 
            indices, 
            0,
            -20,
            Matrix44.from_matrix33(
                drone.coordinate_system.rotation),
            Matrix44.from_translation(
                Vector3(drone.coordinate_system.origin)))

    print(drone)
    print("frame_count: " + str(frame_count))

run()
end()
