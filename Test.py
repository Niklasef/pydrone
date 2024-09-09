import numpy as np
import logging
import time
import sys
import tensorflow as tf
from DroneEnv import DroneEnv
from Navigation import NavPoint
from tensorflow.keras.models import load_model

def run_test(stop_event, start_nav_point, end_nav_point, gym_input_sim_queue, gym_sim_state_queue, model_file, num_episodes=10):
    # Configure the logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a stream handler to log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)

    # Load the trained model
    model = load_model(model_file)
    max_steps_per_episode = 350

    # Initialize the environment
    env = DroneEnv(start_nav_point, end_nav_point, gym_input_sim_queue, gym_sim_state_queue)
    (optimal_distance, optimal_time) = env.calculate_optimal_distance_and_time()

    for episode in range(num_episodes):
        drone_pos = start_nav_point.position
        drone_distance = 0
        start_time = time.time()  # Start time for the episode
        reached_goal = False

        if stop_event.is_set():
            break
        state = np.array(env.reset())
        done = False
        total_reward = 0
        drone_pos = start_nav_point.position
        time_to_goal = 0
        
        for timestep in range(1, max_steps_per_episode):
            current_time = time.time()
            if stop_event.is_set():
                break
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

            state_next, reward, done, _, drone_pos_ = env.step(action, stop_event)
            drone_distance += np.linalg.norm(drone_pos_ - drone_pos)
            drone_pos = drone_pos_
            state = np.array(state_next)

            total_reward += reward

            # Check if the drone is within 0.1 meters of the end navigation point
            if np.linalg.norm(drone_pos - end_nav_point.position) <= 0.1 and not reached_goal:
                time_to_goal = current_time - start_time
                reached_goal = True


        template = "optimal_distance: {}, drone_distance: {}, optimal_time: {}, time_to_goal: {}, reached_goal: {}"
        logging.info(template.format(optimal_distance, drone_distance, optimal_time, time_to_goal, reached_goal))

        # print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    
    stop_event.set()
