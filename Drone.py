from collections import namedtuple
import numpy as np
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local
<<<<<<< HEAD
from Block import Block, create_block, area_x, area_y, area_z, inertia
=======
from Block import Block, create_block, area_x, area_y, area_z, inertia, width, length
>>>>>>> feature/multi-shape-body
from EngineController import compute_forces
from Engine import init_engine_spec, engine_output
from Physics import Body, Velocity
from SpatialObject import SpatialObject

<<<<<<< HEAD
Drone = namedtuple('Drone', 'spatial_object engine_spec')

def init_drone():
    block = create_block(10.0, 1.0, 1.0)

    body = Body(mass=1.0, shape=block)
=======

Drone = namedtuple(
    'Drone',
    [
        'spatial_objects',
        'engine_spec',
        'coordinate_system',
        'vel',
        'engine_positions'
    ])

def init_drone():
    block = create_block(10.0, 1.0, 1.0)
    block_two = create_block(2.0, 2.0, 2.0)

    body = Body(mass=0.5, shape=block)
>>>>>>> feature/multi-shape-body
    coordinate_system = CoordinateSystem(
        origin=np.zeros(3),
        rotation=np.eye(3))
    vel = Velocity(lin=np.zeros(3), rot=np.zeros(3))
    spatial_object = SpatialObject(
        body,
<<<<<<< HEAD
        coordinate_system,
        vel)
=======
        coordinate_system)

    spatial_object_two = SpatialObject(
        Body(mass=0.5, shape=block_two),
        CoordinateSystem(
            origin=np.zeros(3),
            rotation=np.eye(3)))
>>>>>>> feature/multi-shape-body

    engine_spec = init_engine_spec()

    return Drone(
<<<<<<< HEAD
        spatial_object,
        engine_spec)
=======
        [spatial_object, spatial_object_two],
        engine_spec,
        CoordinateSystem(
            origin=np.zeros(3),
            rotation=np.eye(3)),
        Velocity(
            lin=np.zeros(3),
            rot=np.zeros(3)),
        [
            np.array([-(width(block)/2.0), 0.0, (length(block)/2.0)]),
            np.array([(width(block)/2.0), 0.0, (length(block)/2.0)]),
            np.array([(width(block)/2.0), 0.0, -(length(block)/2.0)]),
            np.array([-(width(block)/2.0), 0.0, -(length(block)/2.0)])
        ])
>>>>>>> feature/multi-shape-body
