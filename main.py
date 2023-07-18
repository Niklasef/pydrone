### TODO: ###
# Simple Manual Sim (1m)
# - keyboard input controlling forces
# - engine module with side forces
# - improved lighting with better normals
# - drone module
# - complex shapes
#_____________________________
# CFD calculated air drags (1-2m)
#_____________________________
# AI learning to fly sim (1-3m)
#_____________________________
# use AI model in DIY mini drone using IoT  (1-2m)
#_____________________________
# drone fleet interaction (2-4m)
#_____________________________



from collections import namedtuple
import time
import os
import numpy as np
from WindowRender import init, render, window_active, end
from CoordinateSystem import CoordinateSystem, rotate, translate, rotate_to_global, rotate_to_local
from pyrr import Matrix44, matrix44, Vector3
from Physics import Body, Force, Velocity, apply_rot_force, apply_trans_force, earth_g_force, lin_air_drag, rot_air_torque
from Geometry import Cube, create_cube, area
from KeyboardControler import poll_keyboard
from EngineController import compute_forces


SpatialObject = namedtuple('SpatialObject', 'body coordinateSystem vel')

# Initialize FPS counter variables
frame_count = 0
start_time = time.time()

cube = create_cube(1.0)

vertices = np.array([
    *cube.left_bottom_inner_corner, 0.0, 0.0, -1.0,
    *cube.right_bottom_inner_corner, 0.0, 0.0, -1.0,
    *cube.right_top_inner_corner, 0.0, 0.0, -1.0,
    *cube.left_top_inner_corner, 0.0, 0.0, -1.0,
    *cube.left_bottom_outer_corner, 0.0, 0.0, -1.0,
    *cube.right_bottom_outer_corner, 0.0, 0.0, -1.0,
    *cube.right_top_outer_corner, 0.0, 0.0, -1.0,
    *cube.left_top_outer_corner, 0.0, 0.0, -1.0
], dtype=np.float32)

indices = np.array([
    0, 1, 2, 2, 3, 0,    # Front face
    1, 5, 6, 6, 2, 1,    # Right face
    7, 6, 5, 5, 4, 7,    # Back face
    4, 0, 3, 3, 7, 4,    # Left face
    4, 5, 1, 1, 0, 4,    # Bottom face
    3, 2, 6, 6, 7, 3     # Top face
], dtype=np.uint32)

def stop():
    return [
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([-0.5, 0.0, 0.5]),
            magnitude=0.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, 0.5]),
            magnitude=0.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, -0.5]),
            magnitude=0.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([-0.5, 0.0, -0.5]),
            magnitude=0.0)]

def yaw():
    return [
        Force(
            dir=np.array([0.0, 0.0, 1.0]),
            pos=np.array([-0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, -0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, 1.0]),
            pos=np.array([-0.5, -0.5, 0.0]),
            magnitude=3.0)]

def forward():
    return [
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([-0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, -0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([-0.5, -0.5, 0.]),
            magnitude=3.0)]

def pitch():
    return [
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([-0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, -1.0]),
            pos=np.array([0.5, 0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, 1.0]),
            pos=np.array([0.5, -0.5, 0.0]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 0.0, 1.0]),
            pos=np.array([-0.5, -0.5, 0.]),
            magnitude=3.0)]

def roll():
    return [
        Force(
            dir=np.array([0.0, -1.0, 0.0]),
            pos=np.array([-0.5, 0.0, -0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, -0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, 0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, -1.0, 0.0]),
            pos=np.array([-0.5, 0.0, 0.5]),
            magnitude=3.0)]

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

def translate_sim(forces, spatialObject, delta_time):
    origin_delta, lin_vel_ = apply_trans_force(
        forces,
        rotate_to_local(
            spatialObject.coordinateSystem,
            spatialObject.vel.lin),
        delta_time,
        spatialObject.body)
    coordinate_system_ = translate(
        spatialObject.coordinateSystem,
        rotate_to_global(spatialObject.coordinateSystem, origin_delta))

    return SpatialObject(
        spatialObject.body,
        coordinate_system_,
        Velocity(
            rotate_to_global(coordinate_system_, lin_vel_),
            spatialObject.vel.rot))

def run():
    frame_count = 0
    body = Body(mass=1.0, cube=cube)
    coordinateSystem = CoordinateSystem(
        origin=np.zeros(3),
        rotation=np.eye(3))
    vel = Velocity(lin=np.zeros(3), rot=np.zeros(3))
    spatialObject = SpatialObject(
        body,
        coordinateSystem,
        vel)
    forces = [
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([-0.5, 0.0, 0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, 0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([0.5, 0.0, -0.5]),
            magnitude=3.0),
        Force(
            dir=np.array([0.0, 1.0, 0.0]),
            pos=np.array([-0.5, 0.0, -0.5]),
            magnitude=3.0)]
    start = time.time()
    time_passed = 0
    prev_frame = 0

    window, shader, VAO = init(vertices, indices)

    while window_active(window):
    # while time_passed < 10.0:
        now = time.time()
        delta_time = now - (prev_frame if prev_frame != 0 else now)
        prev_frame = now

        # os.system('cls')
        # print(spatialObject)
        # print(delta_time)
        # time.sleep(0.002)

        # if time_passed > 12.7:
        #     forces = stop()
        # # yaw right 45
        # if time_passed < 2.6 and time_passed > 2.0:
        #     forces = yaw()
        # if time_passed < 4.6 and time_passed > 2.6:
        #     forces = forward()
        # # pith down 90
        # if time_passed < 5.9 and time_passed > 4.6:
        #     forces = pitch()
        # if time_passed < 7.9 and time_passed > 5.9:
        #     forces = forward()
        # # pith down 90
        # if time_passed < 9.2 and time_passed > 7.9:
        #     forces = pitch()
        # if time_passed < 11.4 and time_passed > 9.2:
        #     forces = forward()
        # #roll left 90 (when upside down)
        # if time_passed < 12.7 and time_passed > 11.4:
        #     forces = roll()
        pressed = poll_keyboard()
        
        engine_forces = compute_forces(
            yaw=pressed['y_rot'],
            pitch=pressed['x_rot'],
            roll=pressed['z_rot'],
            power=pressed['y_trans'],
            engine_max_force=5,
            rot_mat=spatialObject.coordinateSystem.rotation,
            rot_vel=spatialObject.vel.rot,
            mass=spatialObject.body.mass,
            coordinate_system=spatialObject.coordinateSystem)
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

        cube_area = area(spatialObject.body.cube)

        rot_air_torque_ = rot_air_torque(
            spatialObject.vel.rot,
            cube_area)

        spatialObject = rotate_sim(
            [f1, f2, f3, f4],
            spatialObject,
            delta_time,
            rot_air_torque_)

        g = rotate_force_to_local(
            earth_g_force(spatialObject.body.mass, 9.81),
            spatialObject.coordinateSystem)

        lin_air_drag_ = rotate_force_to_local(
            lin_air_drag(
                spatialObject.vel.lin,
                cube_area),
            spatialObject.coordinateSystem)
        
        spatialObject = translate_sim(
            [f1, f2, f3, f4, g, lin_air_drag_],
            spatialObject,
            delta_time)

        render(
            window, 
            shader, 
            VAO, 
            indices, 
            0,
            -20,
            Matrix44.from_matrix33(
                spatialObject.coordinateSystem.rotation),
            Matrix44.from_translation(
                Vector3(spatialObject.coordinateSystem.origin)))



        frame_count += 1
        time_passed = now - start
    print(spatialObject)
    print("time_passed: " + str(time_passed))
    print("fps: " + str(frame_count / 10.0))
    print("air_drag: " + str(lin_air_drag_))

run()
end()
