import numpy as np


class PidController:
    def __init__(
        self,

        kp=200,
        ki=0.1,
        kd=0.01,
        target=0,
        max_vertical_velocity=3.85,

        kp_pitch=1.0,
        ki_pitch=0.01,
        kd_pitch=0.1,
        max_pitch=0.5,

        kp_roll=1.0,
        ki_roll=0.01,
        kd_roll=0.1,
        max_roll=0.5,

        kp_yaw=10.0,
        ki_yaw=0.1,
        kd_yaw=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral_error = 0
        self.previous_error = 0
        self.max_vertical_velocity = max_vertical_velocity

        # Pitch PID parameters
        self.kp_pitch = kp_pitch
        self.ki_pitch = ki_pitch
        self.kd_pitch = kd_pitch
        self.max_pitch = max_pitch
        self.integral_error_pitch = 0
        self.previous_error_pitch = 0

        # Roll PID parameters
        self.kp_roll = kp_roll
        self.ki_roll = ki_roll
        self.kd_roll = kd_roll        
        self.max_roll = max_roll
        self.integral_error_roll = 0
        self.previous_error_roll = 0

        # Yaw params
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw
        self.integral_error_yaw = 0
        self.previous_error_yaw = 0
        self.max_yaw_rate = 2


    def blend_outputs(self, vertical, pitch, roll, yaw):
        # Calculate the base thrust for all motors
        base_thrust = vertical

        # Scale the pitch control to match the motor output range [-0.5, 0.5]
        pitch_adjustment = (pitch - 0.5) * 2.0

        roll_adjustment = (roll - 0.5) * 2.0

        # Apply pitch adjustments to the front and back motors
        front_left_thrust = base_thrust + pitch_adjustment - roll_adjustment
        front_right_thrust = base_thrust + pitch_adjustment + roll_adjustment
        back_right_thrust = base_thrust - pitch_adjustment + roll_adjustment
        back_left_thrust = base_thrust - pitch_adjustment - roll_adjustment

        # Apply yaw adjustments
        front_left_thrust += yaw
        back_right_thrust += yaw
        front_right_thrust -= yaw
        back_left_thrust -= yaw

        # Clip the thrust values to ensure they are within the range [0, 1]
        front_left_thrust = np.clip(front_left_thrust, 0, 1)
        front_right_thrust = np.clip(front_right_thrust, 0, 1)
        back_left_thrust = np.clip(back_left_thrust, 0, 1)
        back_right_thrust = np.clip(back_right_thrust, 0, 1)

        # Return the thrust values for all four motors
        return np.array([front_left_thrust, front_right_thrust, back_right_thrust, back_left_thrust])

    def thrust(self, throttle_input, current_vertical_speed, delta_time):  
        throttle_input = throttle_input * self.max_vertical_velocity
        error = self.target - current_vertical_speed + throttle_input
        self.integral_error += error * delta_time
        derivative_error = (error - self.previous_error) / delta_time
        self.previous_error = error

        pid_output = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

        # Normalize the PID output to be within the range [-max_vertical_velocity, max_vertical_velocity]
        pid_output = np.clip(pid_output, -self.max_vertical_velocity, self.max_vertical_velocity)

        # Convert the PID output to motor outputs
        thrust_output = (pid_output + self.max_vertical_velocity) / (2 * self.max_vertical_velocity)

        # Ensure the motor output is within the range [0, 1]
        thrust_output = np.clip(thrust_output, 0, 1)

        return thrust_output

    def pitch_output(self, pitch_input, current_pitch, delta_time):
        pitch_input = pitch_input * self.max_pitch
        error_pitch = self.target - current_pitch + pitch_input
        self.integral_error_pitch += error_pitch * delta_time
        derivative_error_pitch = (error_pitch - self.previous_error_pitch) / delta_time
        self.previous_error_pitch = error_pitch

        pid_output_pitch = self.kp_pitch * error_pitch + self.ki_pitch * self.integral_error_pitch + self.kd_pitch * derivative_error_pitch
        pid_output_pitch = np.clip(pid_output_pitch, -self.max_pitch, self.max_pitch)
        pitch_output_ = (pid_output_pitch + self.max_pitch) / (2 * self.max_pitch)
        pitch_output_ = np.clip(pitch_output_, 0, 1)

        return pitch_output_

    def roll_output(self, roll_input, current_roll, delta_time):
        roll_input = roll_input * self.max_roll
        error_roll = self.target - current_roll + roll_input
        self.integral_error_roll += error_roll * delta_time
        derivative_error_roll = (error_roll - self.previous_error_roll) / delta_time
        self.previous_error_roll = error_roll

        pid_output_roll = self.kp_roll * error_roll + self.ki_roll * self.integral_error_roll + self.kd_roll * derivative_error_roll
        pid_output_roll = np.clip(pid_output_roll, -self.max_roll, self.max_roll)
        roll_output_ = (pid_output_roll + self.max_roll) / (2 * self.max_roll)
        roll_output_ = np.clip(roll_output_, 0, 1)

        return roll_output_

    def yaw_output(self, yaw_input_rate, current_yaw_rate, delta_time):
        yaw_input_rate = yaw_input_rate * self.max_yaw_rate
        error_yaw = yaw_input_rate - current_yaw_rate
        self.integral_error_yaw += error_yaw * delta_time
        derivative_error_yaw = (error_yaw - self.previous_error_yaw) / delta_time
        self.previous_error_yaw = error_yaw

        pid_output_yaw = self.kp_yaw * error_yaw + self.ki_yaw * self.integral_error_yaw + self.kd_yaw * derivative_error_yaw

        # Assuming max_yaw_rate defines the maximum rate of change of yaw.
        # This should be determined based on your drone's capability.
        pid_output_yaw = np.clip(pid_output_yaw, -self.max_yaw_rate, self.max_yaw_rate)

        return pid_output_yaw


    def compute_forces(
        self,
        throttle_input,
        current_vertical_speed,
        delta_time,
        pitch_input,
        current_pitch,
        roll_input,
        current_roll,
        yaw_input,
        current_yaw_rate):

        thrust_output = self.thrust(
            throttle_input,
            current_vertical_speed,
            delta_time)

        pitch_output_ = self.pitch_output(
            pitch_input,
            current_pitch,
            delta_time)

        roll_output_ = self.roll_output(
            -roll_input,
            -current_roll,
            delta_time)

        yaw_output_ = self.yaw_output(
            yaw_input,
            current_yaw_rate,
            delta_time)


        return self.blend_outputs(
            thrust_output,
            pitch_output_,
            roll_output_,
            yaw_output_,
        )

