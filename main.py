from collections import namedtuple
import time
import os
import numpy as np


Body = namedtuple('Body', 'mass')
Velocity = namedtuple('Velocity', 'lin rot')
SpatialObject = namedtuple('SpatialObject', 'body pos vel')
Force = namedtuple('Force', 'dir magnitude')

def run():
    frame_count = 0
    body = Body(1.0)
    vel = Velocity(lin=np.zeros(3), rot=np.zeros(3))
    spatialObject = SpatialObject(body=body, pos=np.zeros(3), vel=vel)
    f1 = Force(np.array([0.0, 1.0, 0.0]), 4.0)
    start = time.time()
    time_passed = 0
    prev_frame = 0

    while time_passed < 10.0:
        now = time.time()
        delta_time = now - (prev_frame if prev_frame != 0 else now)
        prev_frame = now

        # os.system('cls')
        # print(body)
        # print(delta_time)
        #time.sleep(0.002)
        lin_acc = (f1.magnitude * f1.dir) / spatialObject.body.mass
        lin_vel_ = spatialObject.vel.lin + (lin_acc * delta_time)
        vel_ = Velocity(lin=lin_vel_, rot=spatialObject.vel.rot)
        pos_ = spatialObject.pos + (lin_vel_ * delta_time)
        spatialObject = SpatialObject(body, pos_, vel_)

        frame_count += 1
        time_passed = now - start
    print(spatialObject)
    print("time_passed: " + str(time_passed))
    print("fps: " + str(frame_count / 10.0))

run()
