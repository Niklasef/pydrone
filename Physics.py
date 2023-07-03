import numpy as np
from collections import namedtuple


Body = namedtuple('Body', 'mass')
Force = namedtuple('Force', 'dir pos magnitude')
Velocity = namedtuple('Velocity', 'lin rot')

def apply_rot_force(force, rot_vel, time, body):
    inertia = body.mass * (1.0 * 1.0) * (3.0/10.0) #  when force at corner and acting perpendicular to face

    tourque = np.cross(force.pos, force.dir * force.magnitude)
    rot_acc = tourque / inertia
    rot_vel_delta = rot_acc * time
    rot_vel_ = rot_vel + rot_vel_delta
    rot_speed = np.linalg.norm(rot_vel_)
    rot_axis = rot_vel_ / rot_speed if rot_speed > 0 else np.array([1.0, 0.0, 0.0])
    rot_angle = rot_speed * time

    return rot_axis, rot_angle, rot_vel_

def apply_trans_force(force, lin_vel, time, body):
    lin_acc = (force.magnitude * force.dir) / body.mass
    lin_vel_delta = lin_acc * time
    lin_vel_ = lin_vel + lin_vel_delta
    origin_delta = lin_vel_ * time

    return origin_delta, lin_vel_

