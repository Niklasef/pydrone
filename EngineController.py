import numpy as np
from CoordinateSystem import rotate_to_global, euler_angles
from Physics import EARTH_ACC
from scipy.spatial.transform import Rotation as R


def inner_compute_forces(yaw, pitch, roll, power, engine_max_force, rot_mat, rot_vel, mass, coordinate_system):
    # Compute the force needed to counteract gravity
    gravity_force = mass * EARTH_ACC / 4  # distribute among 4 engines
    # Parameters for PD controller
    k_p = 1.0  # proportional gain
    k_d = 1.5  # derivative gain

    (current_pitch, current_roll, current_yaw) = euler_angles(coordinate_system)
    current_pitch = -current_pitch

    if roll == 0:
        roll_error = 0 - current_roll  # error = desired - current, desired roll is 0 for level flight
        roll_rate = -rot_vel[2]  # rotational velocity around z-axis
        roll_adjustment = k_p * roll_error - k_d * roll_rate
        roll += roll_adjustment

    if pitch == 0:
        pitch_error = 0 - current_pitch  # error = desired - current, desired pitch is 0 for level flight
        pitch_rate = rot_vel[0]  # rotational velocity around y-axis
        pitch_adjustment = k_p * pitch_error - k_d * pitch_rate
        pitch -= pitch_adjustment

    if yaw == 0:
        damping_factor = 10.0
        yaw_rate = -rot_vel[1]  # rotational velocity around y-axis
        yaw_damping = damping_factor * yaw_rate
        yaw -= yaw_damping

    # Compute forces for movement
    thrust_vector = np.array([yaw, pitch, roll])
    movement_force = thrust_vector * engine_max_force

    # Compute additional power force to add or reduce altitude
    power_force = power * engine_max_force

    engine_forces = [
        min(max(gravity_force + movement_force[0] + movement_force[1] + movement_force[2] + power_force, 0), engine_max_force),  # engine 1
        min(max(gravity_force - movement_force[0] + movement_force[1] - movement_force[2] + power_force, 0), engine_max_force),  # engine 2
        min(max(gravity_force + movement_force[0] - movement_force[1] - movement_force[2] + power_force, 0), engine_max_force),  # engine 3
        min(max(gravity_force - movement_force[0] - movement_force[1] + movement_force[2] + power_force, 0), engine_max_force),  # engine 4
    ]

    engine_tot_force = sum((f for f in engine_forces), start=0)
    engine_glob_dir = rotate_to_global(coordinate_system, [0, 1, 0])
    engine_glob_y_tot_force = engine_glob_dir[1] * engine_tot_force

    diff =  engine_glob_y_tot_force - (gravity_force*4.0)

    extra_level_percent = 0.0
    if power == 0 and diff < -0.01 and engine_glob_dir[1] > 0:
        available_engine_force = sum(((engine_max_force-f) if f < engine_max_force else 0 for f in engine_forces), start=0)
        engine_glob_y_available_force = engine_glob_dir[1] * available_engine_force
        extra_level_percent = ((-diff) / engine_glob_y_available_force) if engine_glob_y_available_force > 0 else 0

    return [
        min(max(engine_forces[0], engine_forces[0] + (engine_max_force-engine_forces[0])*extra_level_percent), engine_max_force),
        min(max(engine_forces[1], engine_forces[1] + (engine_max_force-engine_forces[1])*extra_level_percent), engine_max_force),
        min(max(engine_forces[2], engine_forces[2] + (engine_max_force-engine_forces[2])*extra_level_percent), engine_max_force),
        min(max(engine_forces[3], engine_forces[3] + (engine_max_force-engine_forces[3])*extra_level_percent), engine_max_force)
    ]    

def compute_forces(yaw, pitch, roll, power, engine_max_force, rot_mat, rot_vel, mass, coordinate_system):
    engine_forces = inner_compute_forces(yaw, pitch, roll, power, engine_max_force, rot_mat, rot_vel, mass, coordinate_system)
    gravity_force = mass * EARTH_ACC
    engine_tot_force = sum((f for f in engine_forces), start=0)
    engine_glob_dir = rotate_to_global(coordinate_system, [0, 1, 0])
    engine_glob_y_tot_force = engine_glob_dir[1] * engine_tot_force

    diff =  engine_glob_y_tot_force - gravity_force
    if power == 0 and diff < -0.01 and engine_glob_dir[1] > 0:
        engine_forces = inner_compute_forces(0, 0, 0, 1, engine_max_force, rot_mat, rot_vel, mass, coordinate_system)

    return np.array([force / engine_max_force for force in engine_forces])
