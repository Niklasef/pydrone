### TODO: ###
# Simple Manual Sim (1m)
# X- keyboard input controlling forces
# X- refactor Sim Module
# X- engine module with side forces
# X- rectangle shape
# X- drone module
# X- multi body drone
# X- rotatable multi body shapes
# X- engines distrubuted across multiple shapes
# X- HUD with metrics
# X- improved lighting with better normals, different colors for each side
#_____________________________
# X- xbox controller input
# X- PID angle mode
# X- ramp up time for motors (1s)
# X- self contained executable 
# (Unit tests, (controll assistant: dissaster recovery, auto hover/position, self-level mode, horizon mode, acro mode), winds, complex detailed shapes, refactor force type to be single vector not split in magnitude - or possible easy to convert between these two forms? maybe force module?)
#_____________________________
# AI learning to fly sim (1-3m)
# X- nav points
# X- gamify route: off route, time, destination reached 
# - reinforcment learning, create env with sim step
# - routes
#_____________________________
# CFD calculated air drags (1-2m)
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
from CoordinateSystem import CoordinateSystem, transform_to_global, rotate, translate, rotate_to_global, rotate_to_local, euler_angles
from pyrr import Matrix44, matrix44, Vector3
from Physics import Body, Force, Velocity, apply_rot_force, apply_trans_force, earth_g_force, lin_air_drag, rot_air_torque
from Cube import Cube, create_cube, area
from KeyboardController import poll_keyboard
from GamepadController import XboxController
from Sim import init_sim, step_sim, SpatialObject
from Navigation import NavPoint, nav_error


def vertices_indices(drone):
    vertices_list = []
    indices_list = []
    face_indices = [
        0, 1, 2, 2, 3, 0,    # Front face
        1, 5, 6, 6, 2, 1,    # Right face
        7, 6, 5, 5, 4, 7,    # Back face
        4, 0, 3, 3, 7, 4,    # Left face
        4, 5, 1, 1, 0, 4,    # Bottom face
        3, 2, 6, 6, 7, 3     # Top face
    ]
    
    i = 0

    for _, spatial_object in enumerate(drone.spatial_objects):
        corners = [
            spatial_object.body.shape.left_bottom_inner_corner,
            spatial_object.body.shape.right_bottom_inner_corner,
            spatial_object.body.shape.right_top_inner_corner,
            spatial_object.body.shape.left_top_inner_corner,
            spatial_object.body.shape.left_bottom_outer_corner,
            spatial_object.body.shape.right_bottom_outer_corner,
            spatial_object.body.shape.right_top_outer_corner,
            spatial_object.body.shape.left_top_outer_corner,
        ]
        for corner in corners:
            vertices_list.extend(
                transform_to_global(spatial_object.coordinateSystem, corner))
            vertices_list.extend([0.0, 0.0, -1.0])
            if i == 0:
                vertices_list.extend([1.0, 0.0, 0.0])  # Red for first part
            elif i == 1:
                vertices_list.extend([0.0, 1.0, 0.0])  # Green for second part
            elif i == 2:
                vertices_list.extend([1.0, 1.0, 1.0])  # Green for second part
            elif i == 3:
                vertices_list.extend([0.0, 1.0, 0.0])  # Green for second part

        indices_list.extend([index + i*8 for index in face_indices])
        i += 1

    vertices = np.array(vertices_list, dtype=np.float32)
    indices = np.array(indices_list, dtype=np.uint32)

    return vertices, indices

def static_vertices_indices(nav_points):
    vertices_list = []
    indices_list = []
    face_indices = [
        0, 1, 2, 2, 3, 0,    # Front face
        1, 5, 6, 6, 2, 1,    # Right face
        7, 6, 5, 5, 4, 7,    # Back face
        4, 0, 3, 3, 7, 4,    # Left face
        4, 5, 1, 1, 0, 4,    # Bottom face
        3, 2, 6, 6, 7, 3     # Top face
    ]
    
    i = 0
    size = 0.1

    for _, nav_point in enumerate(nav_points):
        corners = [
            np.array([-size,-size,-size] + nav_point.position),
            np.array([size,-size,-size] + nav_point.position),
            np.array([-size,size,-size] + nav_point.position),
            np.array([size,size,-size] + nav_point.position),
            np.array([-size,-size,size] + nav_point.position),
            np.array([size,-size,size] + nav_point.position),
            np.array([-size,size,size] + nav_point.position),
            np.array([size,size,size] + nav_point.position),
        ]
        for corner in corners:
            vertices_list.extend(
                transform_to_global(nav_point.coordinate_system, corner))
            vertices_list.extend([0.0, 0.0, -1.0]) # Normal
            vertices_list.extend([1.0, 1.0, 1.0])  # White

        indices_list.extend([index + i*8 for index in face_indices])
        i += 1

    vertices = np.array(vertices_list, dtype=np.float32)
    indices = np.array(indices_list, dtype=np.uint32)

    return vertices, indices

def run():
    (frame_count, prev_frame, drone, pidController) = init_sim()
    nav_points = [
        NavPoint(
            coordinate_system=CoordinateSystem(
                origin=np.array([0, 0, 0]),
                rotation=np.eye(3)),
            position=np.array([5, 5, 0]))
    ]
    print(nav_points)
    (vertices, indices) = vertices_indices(drone)
    (static_vertices, static_indices) = static_vertices_indices(nav_points)


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
    engine_input = [0, 0, 0, 0]

    window, shader, VAO, box_shader, box_VAO, box_VAO_two, dot_shader, dot_VAO, dot_VBO, static_VAO = init(vertices, indices, static_vertices, static_indices)
    xbox_controller = XboxController()
    nav_error_ = 0
    start_nav_point = NavPoint(
        coordinate_system=CoordinateSystem(
            origin=np.array([0, 0, 0]),
            rotation=np.eye(3)),
        position=np.array([0, 0, 0]))

    while window_active(window):
        gamepad_input = xbox_controller.read()
        input = poll_keyboard()
        (frame_count,
            prev_frame,
            drone,
            pidController,
            engine_input,
            delta_time
        ) = step_sim(
            frame_count,
            prev_frame,
            input,
            drone,
            pidController,
            engine_input)

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
                Vector3(drone.coordinate_system.origin)),
            box_shader,
            box_VAO,
            box_VAO_two,
            dot_shader,
            dot_VAO,
            dot_VBO,
            0.6 + (gamepad_input['z_rot']/10.0),
            -0.8 + (gamepad_input['x_rot']/10.0),
            0.6 + (gamepad_input['y_rot']/10.0),
            -0.8 + (gamepad_input['y_trans']/10.0),
            static_VAO,
            static_indices)

        (nav_error_, nav_goal_reached) = nav_error(
            nav_error_,
            start_nav_point,
            nav_points[0],
            drone,
            delta_time)

        print(nav_error_, nav_goal_reached)

    print(drone)
    print("frame_count: " + str(frame_count))

run()
end()
