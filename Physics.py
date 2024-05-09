import numpy as np
from collections import namedtuple
from CoordinateSystem import rotate_to_local


EARTH_ACC = 9.81  # gravity force

Body = namedtuple('Body', 'mass shape')
Force = namedtuple('Force', 'dir pos magnitude')
Velocity = namedtuple('Velocity', 'lin rot')

def apply_rot_force(local_forces, rot_vel, time, rot_drag_torque, engine_torque, inertia):
    total_torque = sum((np.cross(force.pos, force.dir * force.magnitude) for force in local_forces), start=np.array([0.0, 0.0, 0.0]))
    total_torque += rot_drag_torque
    total_torque += engine_torque
    rot_acc = total_torque / inertia

    rot_vel_delta = rot_acc * time
    rot_vel_ = rot_vel + rot_vel_delta
    rot_speed = np.linalg.norm(rot_vel_)

    rot_axis = rot_vel_ / rot_speed if rot_speed > 0 else np.array([1.0, 0.0, 0.0])
    rot_angle = rot_speed * time

    return rot_axis, rot_angle, rot_vel_

def apply_trans_force(local_forces, lin_vel, time, mass):
    total_force = sum((force.magnitude * force.dir for force in local_forces), start=0)
    lin_acc = total_force / mass
    lin_vel_delta = lin_acc * time
    lin_vel_ = lin_vel + lin_vel_delta
    origin_delta = lin_vel_ * time

    return origin_delta, lin_vel_

def earth_g_force(body_mass, acc=EARTH_ACC):
    return Force(
        dir=np.array([0.0, -1.0, 0.0]),
        pos=np.array([0.0, 0.0, 0.0]),
        magnitude=acc * body_mass)

def rotate_force_to_local(force, coordinate_system):
    return Force(
        rotate_to_local(coordinate_system, force.dir),
        force.pos,
        force.magnitude)

def local_g_force(local_coordinate_system, mass):
    return rotate_force_to_local(
        earth_g_force(
            mass,
            9.81),
        local_coordinate_system)

def lin_air_drag(lin_vel, area_x, area_y, area_z, drag_multiplier=1.0):
    # C_d = 1.1 # for cube
    C_d_short_side = 1.0
    C_d_long_side = 2.1
    rho = 1.225

    F_x = 0.5 * C_d_long_side * area_x * rho * lin_vel[0]**2 * drag_multiplier
    F_y = 0.5 * C_d_long_side * area_y * rho * lin_vel[1]**2 * drag_multiplier
    F_z = 0.5 * C_d_short_side * area_z * rho * lin_vel[2]**2 * drag_multiplier

    F_x = -F_x if lin_vel[0] > 0 else F_x
    F_y = -F_y if lin_vel[1] > 0 else F_y
    F_z = -F_z if lin_vel[2] > 0 else F_z
    magnitude = np.linalg.norm([F_x, F_y, F_z])

    return Force(
        dir=np.array([F_x, F_y, F_z])/(magnitude if magnitude > 0.0 else 1.0),
        pos=np.array([0.0, 0.0, 0.0]),
        magnitude=magnitude)


def rot_air_torque(rot_vel, area, drag_multiplier=1.0):
    C_d_rot = 0.1  # Coefficient of rotational drag, you might need to adjust this
    
    rot_speed = np.linalg.norm(rot_vel)
    rot_drag_magnitude = 0.5 * C_d_rot * area * rot_speed**2 * drag_multiplier

    return -rot_vel / (rot_speed if rot_speed > 0 else 1) * rot_drag_magnitude

