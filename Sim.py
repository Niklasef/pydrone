from collections import namedtuple
import time
import os
import numpy as np
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local
from pyrr import Matrix44, matrix44, Vector3
from Physics import Body, Force, Velocity, apply_rot_force, apply_trans_force, earth_g_force, lin_air_drag, rot_air_torque
from Block import Block, create_block, area_x, area_y, area_z, inertia
from EngineController import compute_forces
from Engine import init_engine_spec, engine_output
from Drone import Drone, init_drone
from SpatialObject import SpatialObject


def init_sim():
    frame_count = 0
    prev_frame = 0
    delta_time = 0

    return (frame_count, prev_frame, init_drone())

def rotate_force_to_local(force, coordinate_system):
    return Force(
        rotate_to_local(coordinate_system, force.dir),
        force.pos,
        force.magnitude)

def rotate_sim(forces, spatialObject, delta_time, rot_air_torque, engine_torque):
    I = inertia(
        spatialObject.body.shape,
        spatialObject.body.mass)

    rot_axis, rot_angle, rot_vel_ = apply_rot_force(
        forces,
        spatialObject.vel.rot,
        delta_time,
        spatialObject.body,
        rot_air_torque,
        np.array([0, engine_torque, 0]),
        I)
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

def step_sim(frame_count, prev_frame, imput, drone):
    now = time.time()
    delta_time = now - (prev_frame if prev_frame != 0 else now)
    prev_frame = now
    
    engine_input = compute_forces(
        yaw=imput['y_rot'],
        pitch=imput['x_rot'],
        roll=imput['z_rot'],
        power=imput['y_trans'],
        engine_max_force=drone.engine_spec.force_curve[-1][1],
        rot_mat=drone.spatial_object.coordinateSystem.rotation,
        rot_vel=drone.spatial_object.vel.rot,
        mass=drone.spatial_object.body.mass,
        coordinate_system=drone.spatial_object.coordinateSystem)
    engine_forces = engine_output(drone.engine_spec, engine_input)
    f1 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([-0.5, 0.0, 0.5]),
        magnitude=engine_forces[0][0])
    f2 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([0.5, 0.0, 0.5]),
        magnitude=engine_forces[1][0])
    f3 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([0.5, 0.0, -0.5]),
        magnitude=engine_forces[2][0])
    f4 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=np.array([-0.5, 0.0, -0.5]),
        magnitude=engine_forces[3][0])

    shape_area_x = area_x(drone.spatial_object.body.shape)
    shape_area_y = area_y(drone.spatial_object.body.shape)
    shape_area_z = area_z(drone.spatial_object.body.shape)

    rot_air_torque_ = rot_air_torque(
        drone.spatial_object.vel.rot,
        shape_area_x)
    spatial_object_ = rotate_sim(
        [f1, f2, f3, f4],
        drone.spatial_object,
        delta_time,
        rot_air_torque_,
        engine_forces[0][1] + engine_forces[1][1] + engine_forces[2][1] + engine_forces[3][1])

    g = rotate_force_to_local(
        earth_g_force(spatial_object_.body.mass, 9.81),
        spatial_object_.coordinateSystem)

    lin_air_drag_ = lin_air_drag(
        rotate_to_local(
            drone.spatial_object.coordinateSystem,
            spatial_object_.vel.lin),
        shape_area_x,
        shape_area_y,
        shape_area_z)
    
    spatial_object_ = translate_sim(
        [f1, f2, f3, f4, g, lin_air_drag_],
        spatial_object_,
        delta_time)

    drone_ = Drone(
        spatial_object_,
        drone.engine_spec)

    return (
        frame_count+1,
        prev_frame,
        drone_)
