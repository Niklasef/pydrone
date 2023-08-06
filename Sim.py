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

<<<<<<< HEAD
def rotate_sim(forces, spatialObject, delta_time, rot_air_torque, engine_torque):
    I = inertia(
        spatialObject.body.shape,
        spatialObject.body.mass)
=======
def rotate_sim(
    forces, 
    drone, 
    delta_time, 
    engine_torque):

    rot_air_torques = [
        rot_air_torque(
            rotate_to_local(
                so.coordinateSystem,
                drone.vel.rot),
            area_x(so.body.shape))
        for so
        in drone.spatial_objects]

    total_rot_air_torque = sum(rot_air_torques)

    inertias = [(
        rotate_to_global(
                so.coordinateSystem,
            inertia(
                so.body.shape,
                so.body.mass)))
        for so
        in drone.spatial_objects]

    I_combined = sum(inertias)
>>>>>>> feature/multi-shape-body

    rot_axis, rot_angle, rot_vel_ = apply_rot_force(
        forces,
        drone.vel.rot,
        delta_time,
<<<<<<< HEAD
        spatialObject.body,
        rot_air_torque,
        np.array([0, engine_torque, 0]),
        I)
=======
        total_rot_air_torque,
        np.array([0, engine_torque, 0]),
        I_combined)

>>>>>>> feature/multi-shape-body
    coordinate_system_ = rotate(
        drone.coordinate_system,
        rotate_to_global(
            drone.coordinate_system,
            rot_axis),
        rot_angle)

    return Drone(
        drone.spatial_objects,
        drone.engine_spec,
        coordinate_system_,
        Velocity(
            drone.vel.lin,
            rot_vel_),
        drone.engine_positions)

def translate_sim(forces, drone, delta_time):
    drone_local_lin_vel = rotate_to_local(
        drone.coordinate_system,
        drone.vel.lin)

    total_mass = sum(
        (so.body.mass
            for so
            in drone.spatial_objects),
        start=0.0)

    g = rotate_force_to_local(
        earth_g_force(
            total_mass,
            9.81),
        drone.coordinate_system)

    lin_air_drags = [(lin_air_drag(
        rotate_to_local(
            so.coordinateSystem,
            drone_local_lin_vel),
        area_x(so.body.shape),
        area_y(so.body.shape),
        area_z(so.body.shape))) for so in drone.spatial_objects]

    origin_delta, lin_vel_ = apply_trans_force(
        [*forces, *lin_air_drags, g],
        drone_local_lin_vel,
        delta_time,
        total_mass)
    coordinate_system_ = translate(
        drone.coordinate_system,
        rotate_to_global(drone.coordinate_system, origin_delta))

    return Drone(
        drone.spatial_objects,
        drone.engine_spec,
        coordinate_system_,
        Velocity(
            rotate_to_global(coordinate_system_, lin_vel_),
            drone.vel.rot),
        drone.engine_positions)

<<<<<<< HEAD
def step_sim(frame_count, prev_frame, imput, drone):
=======
def engine_output_sim(drone, input):
    total_mass = sum(
        (so.body.mass
            for so
            in drone.spatial_objects),
        start=0.0)

    engine_input = compute_forces(
        yaw=input['y_rot'],
        pitch=input['x_rot'],
        roll=input['z_rot'],
        power=input['y_trans'],
        engine_max_force=drone.engine_spec.force_curve[-1][1],
        rot_mat=drone.coordinate_system.rotation,
        rot_vel=drone.vel.rot,
        mass=total_mass,
        coordinate_system=drone.coordinate_system)
    engine_forces = engine_output(drone.engine_spec, engine_input)
    f1 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=drone.engine_positions[0],
        magnitude=engine_forces[0][0])
    f2 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=drone.engine_positions[1],
        magnitude=engine_forces[1][0])
    f3 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=drone.engine_positions[2],
        magnitude=engine_forces[2][0])
    f4 = Force(
        dir=np.array([0.0, 1.0, 0.0]),
        pos=drone.engine_positions[3],
        magnitude=engine_forces[3][0])

    return (
        [f1, f2, f3, f4],
        engine_forces[0][1]
            + engine_forces[1][1]
            + engine_forces[2][1]
            + engine_forces[3][1])

def step_sim(frame_count, prev_frame, input, drone):
>>>>>>> feature/multi-shape-body
    now = time.time()
    delta_time = now - (prev_frame if prev_frame != 0 else now)
    prev_frame = now
    
<<<<<<< HEAD
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

=======
    (engine_forces, engine_torque) = engine_output_sim(drone, input)

    drone_ = rotate_sim(
        engine_forces,
        drone,
        delta_time,
        engine_torque)
    
    drone_ = translate_sim(
        engine_forces,
        drone_,
        delta_time)

>>>>>>> feature/multi-shape-body
    return (
        frame_count+1,
        prev_frame,
        drone_)
