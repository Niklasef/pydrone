import numpy as np
from collections import namedtuple


Body = namedtuple('Body', 'mass')
Force = namedtuple('Force', 'dir pos magnitude')
Velocity = namedtuple('Velocity', 'lin rot')

def apply_rot_force(local_forces, rot_vel, time, body):
    inertia = body.mass * (1.0 * 1.0) * (3.0/10.0) #  when force at corner and acting perpendicular to face
    total_torque = sum((np.cross(force.pos, force.dir * force.magnitude) for force in local_forces), start=np.array([0.0, 0.0, 0.0]))
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
