from collections import namedtuple
import time
import os
import numpy as np


Body = namedtuple('Body', 'pos vel mass')
Force = namedtuple('Force', 'dir magnitude')

def run():
    frame_count = 0
    body = Body(np.zeros(3), np.zeros(3), 1.0)
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
        time.sleep(0.002)
        acc = (f1.magnitude * f1.dir) / body.mass
        vel = body.vel + (acc * delta_time)
        pos = body.pos + (vel * delta_time)
        body = Body(pos, vel, body.mass)

        frame_count += 1
        time_passed = now - start
    print(body)
    print("time_passed: " + str(time_passed))
    print("fps: " + str(frame_count / 10.0))

run()
