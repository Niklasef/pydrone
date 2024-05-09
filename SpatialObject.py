from collections import namedtuple
import numpy as np
from Physics import Force, rot_air_torque, lin_air_drag
from CoordinateSystem import rotate_to_global, rotate_to_local
from Block import area_x, area_y, area_z, inertia


SpatialObject = namedtuple('SpatialObject', 'body coordinateSystem')

def global_rot_air_torque(spatial_objects, global_rot_vel):
    rot_air_torques = [
        rotate_to_global(
            so.coordinateSystem,
            rot_air_torque(
                rotate_to_local(
                    so.coordinateSystem,
                    global_rot_vel),
                area_x(so.body.shape)))
        for so
        in spatial_objects]
    return sum(rot_air_torques)

def global_inertia(spatial_objects):
    inertias = [(
        np.abs(
            rotate_to_global(
                    so.coordinateSystem,
                inertia(
                    so.body.shape,
                    so.body.mass))))
        for so
        in spatial_objects]

    return sum(inertias)

def lin_air_drags(spatial_objects, local_lin_vel):
    lin_air_drags_coords = [(
        lin_air_drag(
            rotate_to_local(
                so.coordinateSystem,
                local_lin_vel),
            area_x(so.body.shape),
            area_y(so.body.shape),
            area_z(so.body.shape)),
        so.coordinateSystem)
        for so
        in spatial_objects]

    return [
        Force(
            rotate_to_global(x[1], x[0].dir),
            x[0].pos,
            x[0].magnitude)
        for x
        in lin_air_drags_coords]

def mass(spatial_objects):
    return sum(
        (so.body.mass
            for so
            in spatial_objects),
        start=0.0)
