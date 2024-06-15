import time
import numpy as np
from Sim import init_sim, step_sim
from Navigation import nav_error, NavPoint
from CoordinateSystem import CoordinateSystem, transform_to_global
from pyrr import Matrix44, matrix44, Vector3


class DroneEnv:
    def __init__(self, start_nav_point, end_nav_point, gym_input_sim_queue, gym_sim_state_queue):
        # Initialization of the environment, setting up variables, etc.
        self.input = {}
        self.input['x_rot'] = 0
        self.input['z_rot'] = 0
        self.input['y_rot'] = 0
        self.input['y_trans'] = 0
        self.distance_to_nav_point = 0
        self.start_nav_point = start_nav_point
        self.end_nav_point = end_nav_point
        self.gym_input_sim_queue = gym_input_sim_queue
        self.gym_sim_state_queue = gym_sim_state_queue
        self.state = self.reset()

    def to_state(self, drone):
        state_list = []
        state_list.extend(self.end_nav_point.position/5)
        state_list.extend(drone.coordinate_system.origin/10)
        state_list.extend(drone.coordinate_system.rotation.flatten())
        state_list.extend(drone.vel.lin/10)
        state_list.extend(drone.vel.rot/3.14)
        return np.array(state_list, dtype=np.float32)
    
    def step(self, action, stop_event):
        """
        Run one timestep of the environment's dynamics when given an action.
        Returns:
            state (object): Agent's observation of the current environment.
            reward (float): Amount of reward returned after previous action.
            done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # Implement the logic for one step of simulation, e.g.,
        state, reward, done, info = self.simulate(action, stop_event)
        return state, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        reset_input = {
            'z_rot': 0,
            'x_rot': 0,
            'y_rot': 0,
            'y_trans': 0,
            'debug': [3]
        }
        self.gym_input_sim_queue.put_nowait(reset_input)
        while True:
            if self.gym_sim_state_queue.qsize() > 0:
                sim_state = self.gym_sim_state_queue.get()
                break
        end_nav_point_global = transform_to_global(
            self.end_nav_point.coordinate_system,
            self.end_nav_point.position)
        self.distance_to_nav_point = np.linalg.norm(
            sim_state.drone.coordinate_system.origin - end_nav_point_global)
        return self.to_state(sim_state.drone)
    
    def close(self):
        """
        Clean up the environment's resources.
        """
        # Cleanup logic
        pass

    def action_to_input(self, action):
        assert 0 <= action < 9  # Ensure action is within the range of 0 to 80

        # Map each action to a position on a 3x3 grid
        # First, decode the action into two parts for each stick
        action_stick1 = action  # Determines the position for stick 1
        # action_stick2 = action % 9   # Determines the position for stick 2

        # Function to map a 0-8 action to a -1 to 1 range on each axis
        def map_to_axis(value):
            # Map value to one of the nine positions (-1, 0, 1) on both axes
            x = value % 3 - 1  # -1, 0, or 1
            y = value // 3 - 1 # -1, 0, or 1
            return x, y

        # Map actions to axis positions
        y_trans, _ = map_to_axis(action_stick1)
        # y_trans, y_rot = map_to_axis(action_stick2)

        # Construct the input dictionary
        input_dict = {
            'x_rot': 0,
            'z_rot': 0,
            'y_trans': y_trans,
            'y_rot': 0,
            'debug': []
        }

        return input_dict

    def simulate(self, action, stop_event):
        """
        Simulates the environment for one step and returns the new state, reward, 
        whether the episode is done, and additional info.
        """
        self.input = self.action_to_input(action)
        self.gym_input_sim_queue.put_nowait(self.input)
        while True and not stop_event.is_set():
            if self.gym_sim_state_queue.qsize() > 0:
                sim_state = self.gym_sim_state_queue.get()
                break

        (nav_error_, nav_goal_reached, self.distance_to_nav_point) = nav_error(
            self.distance_to_nav_point,
            self.start_nav_point,
            self.end_nav_point,
            sim_state.drone,
            sim_state.delta_time)

        # reward = (drone.coordinate_system.origin[0] - prev_drone.coordinate_system.origin[0])
        reward = nav_error_

        return (self.to_state(sim_state.drone), reward, False, None)
