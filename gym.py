import logging
import sys
from datetime import datetime
import time
import numpy as np
import os
import keyboard
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from DroneEnv import DroneEnv
from CoordinateSystem import CoordinateSystem
from Navigation import NavPoint


def create_q_model(num_actions):
    # Input Layer
    inputs = layers.Input(shape=(21,))
    
    # Hidden Layers
    layer1 = layers.Dense(128, activation='relu')(inputs)
    layer2 = layers.Dense(128, activation='relu')(layer1)

    # Output Layer
    actions = layers.Dense(num_actions, activation='linear')(layer2)
    
    # Create Model
    model = keras.Model(inputs=inputs, outputs=actions)
    
    return model

def run_gym(stop_event, gym_input_sim_queue, gym_sim_state_queue, start_nav_point, end_nav_point):
    # Configure the logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a stream handler to log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)

    ### Init ###
    num_actions = 9

    # Configuration paramaters for the whole setup
    # Setup
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 0.10  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
        epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 128
    # max_steps_per_episode = 10000
    max_steps_per_episode = 150


    ################################################################
    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 5000
    # Number of frames for exploration
    epsilon_greedy_frames = 100000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 1000000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 1000
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    env = DroneEnv(start_nav_point, end_nav_point, gym_input_sim_queue, gym_sim_state_queue)
    (optimal_distance, optimal_time) = env.calculate_optimal_distance_and_time()
    model_file = "./gym-output/current/my_model.h5"

    # Load the entire model if it exists
    if os.path.exists(model_file):
        print(f"Loading existing model from {model_file}")
        model = keras.models.load_model(model_file)  # Load the entire model
        model_target = keras.models.load_model("./gym-output/current/my_model_target.h5")  # Load the entire target model
    else:
        print("No existing model found. Creating a new model.")
        model = create_q_model(num_actions)
        model_target = create_q_model(num_actions)


    exit_training = False  # Initialize the flag
    ### Init ###


    # while True:  # Run until solved
    drone_distances = []
    times_to_goal = []
    goal_reached_count = 0
    episodes_since_target_update = 0
    while not stop_event.is_set():
        state = np.array(env.reset())
        drone_pos = start_nav_point.position
        episode_reward = 0
        drone_distance = 0
        start_time = time.time()  # Start time for the episode
        reached_goal = False

        for timestep in range(1, max_steps_per_episode):
            current_time = time.time()  # Current time at this timestep
            if keyboard.is_pressed('q') or stop_event.is_set():  # Check if 'Q' is pressed
                print("Q pressed, exiting training loop and saving model...")
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0),
                    loss=keras.losses.Huber(),
                    metrics=["accuracy"]  # Add any metrics you need
                )
                model.save(model_file)  # Save your model
                model_target.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0),
                    loss=keras.losses.Huber(),
                    metrics=["accuracy"]  # Add any metrics you need
                )
                model_target.save("./gym-output/current/my_model_target.h5")  # Save the target model
                exit_training = True  # Set the flag
                break  # Break out of the inner loop
            # of the agent in a pop up window.
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
                # action = np.random.choice(65535)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _, drone_pos_ = env.step(action, stop_event)
            drone_distance += np.linalg.norm(drone_pos_ - drone_pos)
            
            drone_pos = drone_pos_
            # print(reward)
            state_next = np.array(state_next)

            episode_reward += reward
            # print(f"episode_reward: '{episode_reward}'")

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample, verbose=0)
                # future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                average_distance = np.mean(drone_distances)
                avg_times_to_goal = np.mean(times_to_goal)
                goal_reached_rate = goal_reached_count / episodes_since_target_update
                template = "running reward: {:.2f} at episode {}, frame count {}, loss: {:.4f}, optimal_distance: {}, avg drone_distances: {}, optimal_time: {}, avg_times_to_goal: {}, goal_reached_rate: {}"
                logging.info(template.format(running_reward, episode_count, frame_count, loss.numpy(), optimal_distance, average_distance, optimal_time, avg_times_to_goal, goal_reached_rate))
                drone_distances = []
                times_to_goal = []
                goal_reached_count = 0
                episodes_since_target_update = 0

            # Check if the drone is within 0.1 meters of the end navigation point
            if np.linalg.norm(drone_pos - end_nav_point.position) <= 0.1 and not reached_goal:
                time_to_goal = current_time - start_time
                times_to_goal.append(time_to_goal)
                reached_goal = True
                goal_reached_count += 1

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

            # # FPS calculation and printing
            # end_time = time.time()
            # time_diff = end_time - start_time
            # if time_diff > 0:
            #     fps = 1.0 / time_diff
            #     print("FPS:", fps)
            # else:
            #     print("Time interval too small to calculate FPS")

        if exit_training:
            break  # Break out of the outer loop if the flag is set

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 10:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        # print(f"episode_reward: '{episode_reward}'")
        # print(f"running_reward: '{running_reward}'")
        # print(f"epsilon: '{epsilon}'")

        episode_count += 1
        episodes_since_target_update += 1
        drone_distances.append(drone_distance)

        if running_reward > 10:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            print(running_reward)
            break
