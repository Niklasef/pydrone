import numpy as np
from collections import namedtuple
from pyrr import matrix33, matrix44, Matrix44


CoordinateSystem = namedtuple('CoordinateSystem', 'origin rotation')

def rotate(coordinate_system, rot_axis, rot_angle):
    rotation_delta = matrix33.create_from_axis_rotation(
        rot_axis,
        rot_angle)
    rotation_ = matrix33.multiply(
        coordinate_system.rotation,
        rotation_delta)
    return CoordinateSystem(
            coordinate_system.origin, 
            rotation_)

def translate(coordinate_system, origin_delta):
    origin_ = coordinate_system.origin + origin_delta
    return CoordinateSystem(
            origin_, 
            coordinate_system.rotation)

def rotate_to_global(coordinate_system, local_vector):
    return matrix44.apply_to_vector(
        Matrix44.from_matrix33(coordinate_system.rotation),
        local_vector)

def rotate_to_local(coordinate_system, global_vector):
    return matrix44.apply_to_vector(
        Matrix44.from_matrix33(np.linalg.inv(coordinate_system.rotation)),
        global_vector)
