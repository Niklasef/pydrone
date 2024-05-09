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
# - reinforcment learning, create env with sim step, set fps
# - routes
#_____________________________
# CFD calculated air drags (1-2m)
#_____________________________
# use AI model in DIY mini drone using IoT  (1-2m)
#_____________________________
# drone fleet interaction (2-4m)
#_____________________________

from collections import namedtuple
import sys
import time
import os
import psutil
from multiprocessing import Process, Queue, Event
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
from Drone import metrics

SimState = namedtuple('SimState', 'delta_time drone engine_forces, engine_torque')


def get_input_method():
    if len(sys.argv) > 1:
        input_method = sys.argv[1].lower()  # Get the second argument
        if input_method == "k":
            return "keyboard"
        elif input_method == "g":
            return "gamepad"
    return "keyboard"  # Default to keyboard if no argument or unrecognized argument

def run_on_specific_core(core):
    pid = os.getpid()
    p = psutil.Process(pid)
    p.cpu_affinity([core])

def ef_metrics(engine_forces, engine_torque):
    metrics = "Engines\n"
    metrics += "1\t\t2\n"  # Engine numbers top
    metrics += f"Force: {engine_forces[0].magnitude:.3f}\tForce: {engine_forces[1].magnitude:.3f}\n"
    metrics += f"Torq: {engine_torque:.6f}\tTorq: {engine_torque:.6f}\n"
    metrics += "4\t\t3\n"  # Engine numbers bottom
    metrics += f"Force: {engine_forces[3].magnitude:.3f}\tForce: {engine_forces[2].magnitude:.3f}\n"
    metrics += f"Torq: {engine_torque:.6f}\tTorq: {engine_torque:.6f}\n"

    return metrics

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

def poll_input(input_sim_queue, input_render_queue, stop_event):
    run_on_specific_core(1)
    prev_frame = 0
    input_method = get_input_method()

    while True:
        while True:
            now = time.perf_counter()
            if prev_frame == 0:
                prev_frame = now  # Set prev_frame to current time for the first iteration
            delta_time = now - prev_frame
            time.sleep(0.001)
            if delta_time >= (1 / 100):
                break
        prev_frame = now  # Update prev_frame for the next iteration

        if input_method == "gamepad":
            input = gamepad_controller.read()
        else:
            input = poll_keyboard()

        input_sim_queue.put(input)
        input_render_queue.put(input)

        # if delta_time > 0:
        #     fps = 1 / delta_time
        #     print(f"Input FPS: {fps:.2f}\n")

def render_proc(sim_state_queue, input_queue):
    run_on_specific_core(2)
    while sim_state_queue.qsize() > 0:
        sim_state = sim_state_queue.get()

    nav_points = [
        NavPoint(
            coordinate_system=CoordinateSystem(
                origin=np.array([0, 0, 0]),
                rotation=np.eye(3)),
            position=np.array([5, 5, 0]))
    ]
    # print(nav_points)
    (vertices, indices) = vertices_indices(sim_state.drone)
    (static_vertices, static_indices) = static_vertices_indices(nav_points)

    window, shader, VAO, box_shader, box_VAO, box_VAO_two, dot_shader, dot_VAO, dot_VBO, static_VAO = init(vertices, indices, static_vertices, static_indices)
    gamepad_controller = XboxController()
    nav_error_ = 0
    start_nav_point = NavPoint(
        coordinate_system=CoordinateSystem(
            origin=np.array([0, 0, 0]),
            rotation=np.eye(3)),
        position=np.array([0, 0, 0]))

    input_data = {
        'z_rot': 0,
        'x_rot': 0,
        'y_rot': 0,
        'y_trans': 0
    }
    prev_frame = 0

    time.sleep(2)

    while True:
        time.sleep(0.001)
        while sim_state_queue.qsize() > 0:
            sim_state = sim_state_queue.get()
        while input_queue.qsize() > 0:
            input_data = input_queue.get()
        render(
            window, 
            shader, 
            VAO, 
            indices, 
            0,
            -20,
            Matrix44.from_matrix33(
                sim_state.drone.coordinate_system.rotation),
            Matrix44.from_translation(
                Vector3(sim_state.drone.coordinate_system.origin)),
            box_shader,
            box_VAO,
            box_VAO_two,
            dot_shader,
            dot_VAO,
            dot_VBO,
            0.6 + (input_data['z_rot']/10.0),
            -0.8 + (input_data['x_rot']/10.0),
            0.6 + (input_data['y_rot']/10.0),
            -0.8 + (input_data['y_trans']/10.0),
            static_VAO,
            static_indices)

        now = time.perf_counter()
        if prev_frame == 0:
            prev_frame = now  # Set prev_frame to current time for the first iteration
        delta_time = now - prev_frame
        prev_frame = now
        # if delta_time > 0:
        #     fps = 1 / delta_time
        #     print(f"Render FPS: {fps:.2f}\n")

        # (nav_error_, nav_goal_reached) = nav_error(
        #     nav_error_,
        #     start_nav_point,
        #     nav_points[0],
        #     drone,
        #     delta_time)

def console_proc(sim_state_queue):
    run_on_specific_core(3)
    prev_frame = 0
    while True:
        while True:
            now = time.perf_counter()
            if prev_frame == 0:
                prev_frame = now
            delta_time = now - prev_frame
            if delta_time >= (1 / 3):
                break
        prev_frame = now  # Update prev_frame for the next iteration
        while sim_state_queue.qsize() > 0:
            sim_state = sim_state_queue.get()
        print(metrics(sim_state.drone))
        print(ef_metrics(sim_state.engine_forces, sim_state.engine_torque))

def run(input_queue, render_sim_state_queue, console_sim_state_queue, stop_event, render_flag=True):
    (frame_count, prev_frame, drone, pidController) = init_sim()

    run_on_specific_core(0)

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

    input_data = {
        'z_rot': 0,
        'x_rot': 0,
        'y_rot': 0,
        'y_trans': 0
    }

    while True:
        if not input_queue.empty():
            input_data = input_queue.get()
        (frame_count,
            prev_frame,
            drone,
            pidController,
            engine_input,
            delta_time,
            engine_forces,
            engine_torque
        ) = step_sim(
            frame_count,
            prev_frame,
            input_data,
            drone,
            pidController,
            engine_input)

        sim_state = SimState(
            delta_time,
            drone,
            engine_forces,
            engine_torque)
        if render_flag:
            render_sim_state_queue.put(sim_state)
        console_sim_state_queue.put(sim_state)

        # if delta_time > 0:
        #     fps = 1 / delta_time
        #     print(f"Sim FPS: {fps:.2f}\n")

    while not stop_event.is_set():  # Loop until stop event is set
        print("stop event")
        stop_event.set()

# Usage:
# To run without rendering:
# python your_script.py --no-render
# To run with rendering:
# python your_script.py

if __name__ == "__main__":
    render_flag = "--no-render" not in sys.argv  # Check if --no-render flag is present

    input_sim_queue = Queue()
    input_render_queue = Queue()
    render_sim_state_queue = Queue()
    console_sim_state_queue = Queue()
    stop_event = Event()  # Create an event to signal the input process to stop

    poll_input_proc = Process(target=poll_input, args=(input_sim_queue, input_render_queue, stop_event))
    sim_process = Process(target=run, args=(input_sim_queue, render_sim_state_queue, console_sim_state_queue, stop_event, render_flag))
    render_process = Process(target=render_proc, args=(render_sim_state_queue, input_render_queue))
    console_process = Process(target=console_proc, args=(console_sim_state_queue,))

    try:
        poll_input_proc.start()
        sim_process.start()
        if render_flag:
            render_process.start()
        console_process.start()
    finally:
        # Ensure that the input polling process is stopped
        stop_event.set()

        # Kill all spawned processes
        kill_all_processes()

        # Join the input polling process (ensure it's properly terminated)
        poll_input_proc.join(2)
        sim_process.join(2)
        if render_flag:
            render_process.join(2)
        console_process.join(2)
