from collections import namedtuple
import time
import os

Body = namedtuple('Body', 'x_pos y_pos z_pos x_vel y_vel z_vel mass')
Force = namedtuple('Force', 'x y z magnitude')


def run():
    frame_count = 0
    body = Body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f1 = Force(0.0, 1.0, 0.0, 4.0)
    start = time.time()
    prev_frame = time.time_ns()
    while time.time() - start < 10:
        os.system('cls')
        print(body)
        delta_time = (time.time_ns() - prev_frame) / 1000000000.0
        prev_frame = time.time_ns()
        print(delta_time)
        time.sleep(0.01)
        x_acc = (f1.magnitude * f1.x) / body.mass
        y_acc = (f1.magnitude * f1.y) / body.mass
        z_acc = (f1.magnitude * f1.z) / body.mass
        x_vel = body.x_vel + (x_acc * delta_time)
        y_vel = body.y_vel + (y_acc * delta_time)
        z_vel = body.z_vel + (z_acc * delta_time)
        x_pos = body.x_pos + (body.x_vel * delta_time)
        y_pos = body.y_pos + (body.y_vel * delta_time)
        z_pos = body.z_pos + (body.z_vel * delta_time)
        body = Body(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, body.mass)
        frame_count += 1
    print("fps: " + str(frame_count / 10.0))

run()
