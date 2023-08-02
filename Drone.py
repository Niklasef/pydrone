from collections import namedtuple
import numpy as np
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local
from Block import Block, create_block, area_x, area_y, area_z, inertia
from EngineController import compute_forces
from Engine import init_engine_spec, engine_output
from Physics import Body, Velocity
from SpatialObject import SpatialObject

Drone = namedtuple('Drone', 'spatial_object engine_spec')

def init_drone():
    block = create_block(10.0, 1.0, 1.0)

    body = Body(mass=1.0, shape=block)
    coordinate_system = CoordinateSystem(
        origin=np.zeros(3),
        rotation=np.eye(3))
    vel = Velocity(lin=np.zeros(3), rot=np.zeros(3))
    spatial_object = SpatialObject(
        body,
        coordinate_system,
        vel)

    engine_spec = init_engine_spec()

    return Drone(
        spatial_object,
        engine_spec)
