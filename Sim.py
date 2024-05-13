from collections import namedtuple
import time
import os
import numpy as np
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local, euler_angles
from pyrr import Matrix44, matrix44, Vector3
from Physics import Body, Force, Velocity, apply_rot_force, apply_trans_force, local_g_force
from Block import Block, create_block
from PidEngineController import PidController
from Engine import init_engine_spec, engine_output_with_response
from Drone import Drone, init_drone
from SpatialObject import SpatialObject, global_rot_air_torque, global_inertia, lin_air_drags, mass


def init_sim():
    frame_count = 0
    prev_frame = 0
    delta_time = 0

    pidController = PidController()

    return (frame_count, prev_frame, init_drone(), pidController)

def rotate_sim(
    engine_forces, 
    drone, 
    delta_time, 
    engine_torque):

    rot_axis, rot_angle, rot_vel_ = apply_rot_force(
        engine_forces,
        drone.vel.rot,
        delta_time,
        global_rot_air_torque(drone.spatial_objects, drone.vel.rot),
        np.array([0, engine_torque, 0]),
        global_inertia(drone.spatial_objects))

    coordinate_system_ = rotate(
        drone.coordinate_system,
        rotate_to_global(
            drone.coordinate_system,
            rot_axis),
        rot_angle)

    return Drone(
        drone.spatial_objects,
        drone.engine_spec,
        coordinate_system_,
        Velocity(
            drone.vel.lin,
            rot_vel_),
        drone.engine_positions)

def translate_sim(engine_forces, drone, delta_time):
    drone_local_lin_vel = rotate_to_local(
        drone.coordinate_system,
        drone.vel.lin)

    drone_mass = mass(drone.spatial_objects)

    local_forces = [
        *engine_forces,
        *lin_air_drags(drone.spatial_objects, drone_local_lin_vel),
        local_g_force(drone.coordinate_system, drone_mass)
    ]

    origin_delta, lin_vel_ = apply_trans_force(
        local_forces,
        drone_local_lin_vel,
        delta_time,
        drone_mass)
    coordinate_system_ = translate(
        drone.coordinate_system,
        rotate_to_global(drone.coordinate_system, origin_delta))

    return Drone(
        drone.spatial_objects,
        drone.engine_spec,
        coordinate_system_,
        Velocity(
            rotate_to_global(coordinate_system_, lin_vel_),
            drone.vel.rot),
        drone.engine_positions)

def engine_output_sim(
    drone,
    input,
    pidController,
    delta_time,
    prev_engine_input):
    total_mass = sum(
        (so.body.mass
            for so
            in drone.spatial_objects),
        start=0.0)

    euler_angles_ = euler_angles(drone.coordinate_system)
    engine_input = pidController.compute_forces(
        input['y_trans'],
        drone.vel.lin[1],
        delta_time if delta_time > 0 else 0.0001,
        input['x_rot'],
        euler_angles_[0],
        input['z_rot'],
        euler_angles_[1],
        input['y_rot'],
        -drone.vel.rot[1],
        input['debug']
    )
    engine_forces = engine_output_with_response(
        drone.engine_spec,
        prev_engine_input,
        engine_input,
        delta_time)
    f1 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=rotate_to_global(
            drone.spatial_objects[1].coordinateSystem,
            drone.engine_positions[0]),
        magnitude=engine_forces[0][0])
    f2 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=rotate_to_global(
            drone.spatial_objects[0].coordinateSystem,
            drone.engine_positions[1]),
        magnitude=engine_forces[1][0])
    f3 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=rotate_to_global(
            drone.spatial_objects[1].coordinateSystem,
            drone.engine_positions[2]),
        magnitude=engine_forces[2][0])
    f4 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=rotate_to_global(
            drone.spatial_objects[0].coordinateSystem,
            drone.engine_positions[3]),
        magnitude=engine_forces[3][0])

    return (
        [f1, f2, f3, f4],
        engine_forces[0][1]
            + engine_forces[1][1]
            + engine_forces[2][1]
            + engine_forces[3][1],
        pidController,
        engine_input)

def step_sim(frame_count, prev_frame, input, drone, pidController, prev_engine_input):
    while True:
        now = time.perf_counter()
        if prev_frame == 0:
            prev_frame = now  # Set prev_frame to current time for the first iteration
        delta_time = now - prev_frame
        if delta_time >= (1 / 350):
            break

    prev_frame = now  # Update prev_frame for the next iteration

    (engine_forces,
        engine_torque,
        pidController,
        engine_input
    ) = engine_output_sim(
        drone,
        input,
        pidController,
        delta_time,
        prev_engine_input)

    drone_ = rotate_sim(
        engine_forces,
        drone,
        delta_time,
        engine_torque)
    
    drone_ = translate_sim(
        engine_forces,
        drone_,
        delta_time)

    return (
        frame_count+1,
        prev_frame,
        drone_,
        pidController,
        engine_input,
        delta_time,
        engine_forces,
        engine_torque)
