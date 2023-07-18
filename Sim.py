from collections import namedtuple
import time
import os
import numpy as np
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local
from pyrr import Matrix44, matrix44, Vector3
from Physics import Body, Force, Velocity, apply_rot_force, apply_trans_force, earth_g_force, lin_air_drag, rot_air_torque
from Geometry import Cube, create_cube, area
from KeyboardControler import poll_keyboard
from EngineController import compute_forces


SpatialObject = namedtuple('SpatialObject', 'body coordinateSystem vel')

def init_sim():
    frame_count = 0
    prev_frame = 0
    delta_time = 0
    cube = create_cube(1.0)

    body = Body(mass=1.0, cube=cube)
    coordinate_system = CoordinateSystem(
        origin=np.zeros(3),
        rotation=np.eye(3))
    vel = Velocity(lin=np.zeros(3), rot=np.zeros(3))
    spatial_object = SpatialObject(
        body,
        coordinate_system,
        vel)

    return (spatial_object, frame_count, prev_frame)

def rotate_force_to_local(force, coordinate_system):
    return Force(
        rotate_to_local(coordinate_system, force.dir),
        force.pos,
        force.magnitude)

def rotate_sim(forces, spatialObject, delta_time, rot_air_torque):
    cube_area = area(spatialObject.body.cube)

    rot_axis, rot_angle, rot_vel_ = apply_rot_force(
        forces,
        spatialObject.vel.rot,
        delta_time,
        spatialObject.body,
        rot_air_torque,
        cube_area)
    coordinate_system_ = rotate(
        spatialObject.coordinateSystem,
        rotate_to_global(
            spatialObject.coordinateSystem,
            rot_axis),
        rot_angle)

    return SpatialObject(
        spatialObject.body,
        coordinate_system_,
        Velocity(
            spatialObject.vel.lin,
            rot_vel_))

def translate_sim(forces, spatial_object, delta_time):
    origin_delta, lin_vel_ = apply_trans_force(
        forces,
        rotate_to_local(
            spatial_object.coordinateSystem,
            spatial_object.vel.lin),
        delta_time,
        spatial_object.body)
    coordinate_system_ = translate(
        spatial_object.coordinateSystem,
        rotate_to_global(spatial_object.coordinateSystem, origin_delta))

    return SpatialObject(
        spatial_object.body,
        coordinate_system_,
        Velocity(
            rotate_to_global(coordinate_system_, lin_vel_),
            spatial_object.vel.rot))

def step_sim(spatial_object, frame_count, prev_frame):
    now = time.time()
    delta_time = now - (prev_frame if prev_frame != 0 else now)
    prev_frame = now    
    pressed = poll_keyboard()
    
    engine_forces = compute_forces(
        yaw=pressed['y_rot'],
        pitch=pressed['x_rot'],
        roll=pressed['z_rot'],
        power=pressed['y_trans'],
        engine_max_force=5,
        rot_mat=spatial_object.coordinateSystem.rotation,
        rot_vel=spatial_object.vel.rot,
        mass=spatial_object.body.mass,
        coordinate_system=spatial_object.coordinateSystem)
    f1 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([-0.5, 0.0, 0.5]),
        magnitude=engine_forces[0])
    f2 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([0.5, 0.0, 0.5]),
        magnitude=engine_forces[1])
    f3 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([0.5, 0.0, -0.5]),
        magnitude=engine_forces[2])
    f4 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([-0.5, 0.0, -0.5]),
        magnitude=engine_forces[3])

    cube_area = area(spatial_object.body.cube)

    rot_air_torque_ = rot_air_torque(
        spatial_object.vel.rot,
        cube_area)

    spatial_object = rotate_sim(
        [f1, f2, f3, f4],
        spatial_object,
        delta_time,
        rot_air_torque_)

    g = rotate_force_to_local(
        earth_g_force(spatial_object.body.mass, 9.81),
        spatial_object.coordinateSystem)

    lin_air_drag_ = rotate_force_to_local(
        lin_air_drag(
            spatial_object.vel.lin,
            cube_area),
        spatial_object.coordinateSystem)
    
    spatial_object = translate_sim(
        [f1, f2, f3, f4, g, lin_air_drag_],
        spatial_object,
        delta_time)

    return (spatial_object, frame_count+1, prev_frame)
