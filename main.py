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
import argparse
from multiprocessing import Process, Queue, Event
from queue import Empty
import numpy as np
from WindowRender import init, render, window_active, end, static_vertices_indices, vertices_indices
from CoordinateSystem import CoordinateSystem, transform_to_global, rotate, translate, rotate_to_global, rotate_to_local, euler_angles
from pyrr import Matrix44, matrix44, Vector3
from Physics import Body, Force, Velocity, apply_rot_force, apply_trans_force, earth_g_force, lin_air_drag, rot_air_torque
from Cube import Cube, create_cube, area
from KeyboardController import poll_keyboard
from GamepadController import XboxController
from Sim import init_sim, step_sim, SpatialObject
from gym import run_gym
from Navigation import NavPoint, nav_error
from Drone import metrics

SimState = namedtuple('SimState', 'delta_time drone engine_forces, engine_torque, pidController')


def get_input_method():
    if len(sys.argv) > 1:
        input_method = sys.argv[1].lower()  # Get the second argument
        if input_method == "k":
            return "keyboard"
        elif input_method == "g":
            return "gamepad"
    return "keyboard"  # Default to keyboard if no argument or unrecognized argument

def run_on_specific_cores(core_start, core_end):
    pid = os.getpid()
    p = psutil.Process(pid)
    cores = list(range(core_start, core_end + 1))
    p.cpu_affinity(cores)

def ef_metrics(engine_forces, engine_torque):
    metrics = "Engines\n"
    metrics += "1\t\t2\n"  # Engine numbers top
    metrics += f"Force: {engine_forces[0].magnitude:.3f}\tForce: {engine_forces[1].magnitude:.3f}\n"
    metrics += f"Torq: {engine_torque:.6f}\tTorq: {engine_torque:.6f}\n"
    metrics += "4\t\t3\n"  # Engine numbers bottom
    metrics += f"Force: {engine_forces[3].magnitude:.3f}\tForce: {engine_forces[2].magnitude:.3f}\n"
    metrics += f"Torq: {engine_torque:.6f}\tTorq: {engine_torque:.6f}\n"

    return metrics

def pid_metrics(pid):
    metrics = "PID Metrics (Dynamic Values)\n"

    # Vertical control dynamic PID values
    metrics += "Vertical Control:\n"
    metrics += f"  Current Error: {pid.previous_error:.3f}\n"
    metrics += f"  Integral Error: {pid.integral_error:.3f}\n"
    # Derivative error can be calculated manually if needed
    metrics += f"  Derivative Error: {pid.kd * ((pid.target - pid.previous_error) / pid.max_vertical_velocity):.3f}\n\n"

    # Pitch control dynamic PID values
    metrics += "Pitch Control:\n"
    metrics += f"  Current Error: {pid.previous_error_pitch:.3f}\n"
    metrics += f"  Integral Error: {pid.integral_error_pitch:.3f}\n"
    metrics += f"  Derivative Error: {pid.kd_pitch * ((pid.target - pid.previous_error_pitch) / pid.max_pitch):.3f}\n\n"

    # Roll control dynamic PID values
    metrics += "Roll Control:\n"
    metrics += f"  Current Error: {pid.previous_error_roll:.3f}\n"
    metrics += f"  Integral Error: {pid.integral_error_roll:.3f}\n"
    metrics += f"  Derivative Error: {pid.kd_roll * ((pid.target - pid.previous_error_roll) / pid.max_roll):.3f}\n\n"

    # Yaw control dynamic PID values
    metrics += "Yaw Control:\n"
    metrics += f"  Current Error: {pid.previous_error_yaw:.3f}\n"
    metrics += f"  Integral Error: {pid.integral_error_yaw:.3f}\n"
    metrics += f"  Derivative Error: {pid.kd_yaw * ((pid.target - pid.previous_error_yaw) / pid.max_yaw_rate):.3f}\n"

    return metrics

def poll_input(input_sim_queue, input_render_queue, stop_event, render_flag, gym_flag):
    run_on_specific_cores(1, 1)
    prev_frame = 0
    input_method = get_input_method()
    print('Press: 1 - quit, 0 - reset PID errors')

    while not stop_event.is_set():
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
        if 1 in input['debug']:
            print("Quit command received. Setting stop event.")
            stop_event.set()
            print("stop event set")

        if not gym_flag:
            input_sim_queue.put_nowait(input)
        if render_flag:
            input_render_queue.put_nowait(input)


        # if delta_time > 0:
        #     fps = 1 / delta_time
        #     print(f"Input FPS: {fps:.2f}\n")

def render_proc(sim_state_queue, input_queue, stop_event, nav_points):
    run_on_specific_cores(2, 2)
    while sim_state_queue.qsize() > 0:
        sim_state = sim_state_queue.get()

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

    while not stop_event.is_set():
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

def console_proc(sim_state_queue, stop_event):
    run_on_specific_cores(3, 3)
    prev_frame = 0
    while not stop_event.is_set():
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
        print(pid_metrics(sim_state.pidController))

        if sim_state.delta_time > 0:
            fps = 1 / sim_state.delta_time
            print(f"Sim FPS: {fps:.2f}\n")

def run(input_queue, render_sim_state_queue, console_sim_state_queue, stop_event, console_flag, gym_flag, gym_sim_state_queue, gym_input_sim_queue, render_flag=True):
    run_on_specific_cores(0, 0)

    (frame_count, prev_frame, drone, pidController) = init_sim()
    start = time.time()
    time_passed = 0
    prev_frame = 0
    engine_input = [0, 0, 0, 0]
    input_data = {
        'z_rot': 0,
        'x_rot': 0,
        'y_rot': 0,
        'y_trans': 0,
        'debug': []
    }

    while not stop_event.is_set():
        gym_input_recieved = False
        if not gym_flag and not input_queue.empty():
            try:
                input_data = input_queue.get_nowait()
            except Empty:
                input_data = {
                    'z_rot': 0,
                    'x_rot': 0,
                    'y_rot': 0,
                    'y_trans': 0,
                    'debug': []
                }
                pass
        elif gym_flag:
            try:
                input_data = gym_input_sim_queue.get_nowait()
                gym_input_recieved = True
            except Empty:
                gym_input_recieved = False
                pass

        # Reset Sim
        if 3 in input_data["debug"]:
            (frame_count, prev_frame, drone, pidController) = init_sim()
            start = time.time()
            time_passed = 0
            prev_frame = 0
            engine_input = [0, 0, 0, 0]
            input_data = {
                'z_rot': 0,
                'x_rot': 0,
                'y_rot': 0,
                'y_trans': 0,
                'debug': []
            }

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
            engine_torque,
            pidController)
        if render_flag:
            render_sim_state_queue.put_nowait(sim_state)
        if console_flag:
            console_sim_state_queue.put_nowait(sim_state)
        if gym_flag and gym_input_recieved:
            gym_sim_state_queue.put_nowait(sim_state)

        # if delta_time > 0:
        #     fps = 1 / delta_time
        #     print(f"Sim FPS: {fps:.2f}\n")

def gym_proc(gym_input_sim_queue, gym_sim_state_queue, stop_event, nav_points):
    run_on_specific_cores(1, 15)
    run_gym(stop_event, gym_input_sim_queue, gym_sim_state_queue, nav_points[0], nav_points[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run drone simulation.')
    parser.add_argument('--no-render', action='store_true', help='Run without rendering.')
    parser.add_argument('--gym', action='store_true', help='Run in gym mode.')
    parser.add_argument('--console', action='store_true', help='Console logging mode.')
    parser.add_argument('--gym-time', type=int, default=75, help='Time in seconds to run the gym simulation.')
    
    args = parser.parse_args()
    
    render_flag = not args.no_render  # Check if --no-render flag is present
    gym_flag = args.gym
    console_flag = args.console
    gym_time = args.gym_time  # Get the gym time from command-line argument

    nav_points = [
        NavPoint(
            coordinate_system=CoordinateSystem(
                origin=np.array([0, 0, 0]),
                rotation=np.eye(3)),
            position=np.array([0, 0, 0])),
        NavPoint(
            coordinate_system=CoordinateSystem(
                origin=np.array([0, 0, 0]),
                rotation=np.eye(3)),
            position=np.array([0, 7, 0]))
    ]

    input_sim_queue = Queue()
    input_render_queue = Queue()
    gym_input_sim_queue = Queue()
    render_sim_state_queue = Queue()
    console_sim_state_queue = Queue()
    gym_sim_state_queue = Queue()
    stop_event = Event()  # Create an event to signal the input process to stop

    poll_input_proc = Process(target=poll_input, args=(input_sim_queue, input_render_queue, stop_event, render_flag, gym_flag))
    sim_process = Process(target=run, args=(input_sim_queue, render_sim_state_queue, console_sim_state_queue, stop_event, console_flag, gym_flag, gym_sim_state_queue, gym_input_sim_queue, render_flag))
    if console_flag:
        console_process = Process(target=console_proc, args=(console_sim_state_queue, stop_event))
    if render_flag:
        render_process = Process(target=render_proc, args=(render_sim_state_queue, input_render_queue, stop_event, nav_points))

    sim_process.start()
    poll_input_proc.start()
    time.sleep(1)
    if render_flag:
        render_process.start()
    if console_flag:
        console_process.start()
    if gym_flag:
        gym_process = Process(target=gym_proc, args=(gym_input_sim_queue, gym_sim_state_queue, stop_event, nav_points))
        gym_process.start()

    time.sleep(gym_time)  # Use the gym time from command-line argument
    stop_event.set()

    # Join the input polling process (ensure it's properly terminated)
    poll_input_proc.join()
    print('poll_input_proc.join()')
    if gym_flag:
        gym_process.join()
        print('gym_process.join()')
    sim_process.join()
    print('sim_process.join()')
    if console_flag:
        console_process.join()
        print('console_process.join()')
    if render_flag:
        render_process.join()
