import numpy as np
from collections import namedtuple


Body = namedtuple('Body', 'mass cube')
Force = namedtuple('Force', 'dir pos magnitude')
Velocity = namedtuple('Velocity', 'lin rot')

def apply_rot_force(local_forces, rot_vel, time, body, rot_drag_torque, area):
    inertia = body.mass * area * (3.0/10.0) #  when force at corner and acting perpendicular to face
    total_torque = sum((np.cross(force.pos, force.dir * force.magnitude) for force in local_forces), start=np.array([0.0, 0.0, 0.0]))
    total_torque += rot_drag_torque
    rot_acc = total_torque / inertia

    rot_vel_delta = rot_acc * time
    rot_vel_ = rot_vel + rot_vel_delta
    rot_speed = np.linalg.norm(rot_vel_)

    rot_axis = rot_vel_ / rot_speed if rot_speed > 0 else np.array([1.0, 0.0, 0.0])
    rot_angle = rot_speed * time

    return rot_axis, rot_angle, rot_vel_

def apply_trans_force(local_forces, lin_vel, time, body):
    total_force = sum((force.magnitude * force.dir for force in local_forces), start=0)
    lin_acc = total_force / body.mass
    lin_vel_delta = lin_acc * time
    lin_vel_ = lin_vel + lin_vel_delta
    origin_delta = lin_vel_ * time

    return origin_delta, lin_vel_

def earth_g_force(body_mass, acc=9.81):
    return Force(
        dir=np.array([0.0, -1.0, 0.0]),
        pos=np.array([0.0, 0.0, 0.0]),
        magnitude=acc * body_mass)

def lin_air_drag(lin_vel, area, drag_multiplier=1.0):
    C_d = 1.1
    rho = 1.225

    V = np.linalg.norm(lin_vel)
    normalized_vel = (-lin_vel) / (V if V > 0 else 1)

    return Force(
        dir=normalized_vel,
        pos=np.array([0.0, 0.0, 0.0]),
        magnitude=0.5 * C_d * area * rho * V**2 * drag_multiplier)

def rot_air_torque(rot_vel, area, drag_multiplier=1.0):
    C_d_rot = 0.1  # Coefficient of rotational drag, you might need to adjust this
    
    rot_speed = np.linalg.norm(rot_vel)
    rot_drag_magnitude = 0.5 * C_d_rot * area * rot_speed**2 * drag_multiplier

    return -rot_vel / (rot_speed if rot_speed > 0 else 1) * rot_drag_magnitude

