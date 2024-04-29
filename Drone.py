from collections import namedtuple
import numpy as np
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local
from Block import Block, create_block, area_x, area_y, area_z, inertia, width, length
from EngineController import compute_forces
from Engine import init_engine_spec, engine_output
from Physics import Body, Velocity
from SpatialObject import SpatialObject


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
    rot_axis = np.array([0, 1, 0])  # Y-axis
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    block = create_block(2.0, 0.1, 0.1)
    coordinate_system = CoordinateSystem(
        origin=np.zeros(3),
        rotation=np.eye(3))
    rot_angle = 45 * np.pi / 180 # Convert 45 degrees to radians
    coordinate_system_ = rotate(
        coordinate_system,
        rot_axis,
        rot_angle)
    spatial_object = SpatialObject(
        Body(mass=0.5, shape=block),
        coordinate_system_)

    block_two = create_block(2.0, 0.1, 0.1)
    coordinate_system_two = CoordinateSystem(
        origin=np.zeros(3),
        rotation=np.eye(3))
    rot_angle = -45 * np.pi / 180 # Convert 45 degrees to radians
    coordinate_system_two_ = rotate(
        coordinate_system_two,
        rot_axis,
        rot_angle)
    spatial_object_two = SpatialObject(
        Body(mass=0.5, shape=block_two),
        coordinate_system_two_)


    block_three = create_block(1.0, 0.05, 0.05)
    coordinate_system_three = CoordinateSystem(
        origin=np.array([0,0,-0.5]),
        rotation=np.eye(3))
    spatial_object_three = SpatialObject(
        Body(mass=0.01, shape=block_three),
        coordinate_system_three)

    block_four = create_block(0.3, 0.3, 0.3)
    coordinate_system_four = CoordinateSystem(
        origin=np.array([0,-0.2,0]),
        rotation=np.eye(3))
    spatial_object_four = SpatialObject(
        Body(mass=0.1, shape=block_four),
        coordinate_system_four)

    engine_spec = init_engine_spec()

    return Drone(
        [spatial_object, spatial_object_two, spatial_object_three, spatial_object_four],
        engine_spec,
        CoordinateSystem(
            origin=np.zeros(3),
            rotation=np.eye(3)),
        Velocity(
            lin=np.zeros(3),
            rot=np.zeros(3)),
        [
            np.array([0.0, 0.0, (length(block)/2.0)]),
            np.array([0.0, 0.0, (length(block_two)/2.0)]),
            np.array([0.0, 0.0, -(length(block)/2.0)]),
            np.array([0.0, 0.0, -(length(block_two)/2.0)])
        ])

def pretty_vector(label, vector):
    if isinstance(vector, np.ndarray):
        # Flatten the array to handle multidimensional arrays properly
        vector = vector.flatten()
    formatted_vector = ", ".join(f"{v:.3f}" for v in vector)
    return f"{label}: [{formatted_vector}]"

def pretty_matrix(label, matrix):
    if isinstance(matrix, np.ndarray):
        # Ensure the matrix is two-dimensional and 3x3
        if matrix.ndim != 2 or matrix.shape[0] != 3 or matrix.shape[1] != 3:
            return "Invalid matrix dimensions"
        # Adjust the format to ensure alignment
        formatted_matrix = "\n  ".join("  ".join(f"{item:6.3f}" for item in row) for row in matrix)
        return f"{label}:\n [{formatted_matrix}]"
    else:
        return "Invalid input: Not a matrix"

def metrics(drone):
    metrics_output = (
        pretty_vector('pos', drone.coordinate_system.origin) + "\n" +
        pretty_matrix('rot', drone.coordinate_system.rotation) + "\n" +
        pretty_vector('lin vel', drone.vel.lin) + "\n" +
        pretty_vector('rot vel', drone.vel.rot) + "\n"
    )
    return metrics_output
