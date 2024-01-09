import time
from collections import deque, namedtuple

import gym
import pygame
import numpy as np
import tensorflow as tf
import os
import utils

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

# Set the random seed for TensorFlow
tf.random.set_seed(utils.SEED)

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class LunarLanderClass:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.state_size = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.completion_average = 100 #The number of attempts needed to be above a certain average to complete training
        
        # Create a Q-Network
        self.q_network = Sequential([
            tf.keras.layers.Input(self.state_size),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.num_actions, activation="linear")
            ])
        
        # Create a target Q^-Network
        self.target_q_network = Sequential([
            tf.keras.layers.Input(self.state_size),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.num_actions, activation="linear")
            ])
        
        self.optimizer = tf.keras.optimizers.Adam(ALPHA)
        
        self.env.reset()

    def OnStart(self):
        self.env.reset()
        pygame.init()

    def OnExit(self):
        self.env.close()

""" 
Calculates the loss.

Args:
    experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
    gamma: (float) The discount factor.
    q_network: (tf.keras.Sequential) Keras model for predicting the q_values
    target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
        
Returns:
    loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
        the y targets and the Q(s,a) values.
"""
def compute_loss(experiences, gamma, q_network, target_q_network):
    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + gamma*max_qsa*(1 - done_vals)

    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))
        
    # Compute the loss
    loss = MSE(y_targets, q_values)  
    return loss

"""
Updates the weights of the Q networks.

Args:
    experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
    gamma: (float) The discount factor.

"""
@tf.function
def agent_learn(experiences, gamma, q_network, target_q_network, optimizer):    
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)


def train_agent(total_point_history, lunarLander: LunarLanderClass, create_videos=0, stepNum=100):
    start = time.time()

    num_episodes = 2000
    max_num_timesteps = 1000

    num_p_av = lunarLander.completion_average    # number of total points to use for averaging
    epsilon = 1.0     # initial ε value for ε-greedy policy

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights
    lunarLander.target_q_network.set_weights(lunarLander.q_network.get_weights())

    filename = ""
    if create_videos:
        filename += "./videos/lunar_lander"

    for i in range(num_episodes):
        
        # Reset the environment to the initial state and get the initial state
        state = lunarLander.env.reset()
        total_points = 0
        
        for t in range(max_num_timesteps):
            
            # From the current state S choose an action A using an ε-greedy policy
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
            q_values = lunarLander.q_network(state_qn)
            action = utils.get_action(q_values, epsilon)
            
            # Take action A and receive reward R and the next state S'
            next_state, reward, done, _ = lunarLander.env.step(action)
            
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))
            
            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
            
            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = utils.get_experiences(memory_buffer)
                
                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                agent_learn(experiences, GAMMA, lunarLander.q_network, lunarLander.target_q_network, lunarLander.optimizer)
            
            state = next_state.copy()
            total_points += reward
            
            if done:
                break
                
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        
        # Update the ε value
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

        if (i+1) % num_p_av == 0:
            print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if av_latest_points >= 200.0:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            lunarLander.q_network.save('lunar_lander_model.h5')
            break

        if create_videos > 0 and (i+1) % stepNum == 0:
            utils.create_videos(filename, lunarLander.env, lunarLander.q_network, create_videos, i+1)
            
    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

def plotPointHistory(total_point_history):
    # Plot the total point history along with the moving average
    utils.plot_history(total_point_history)

def PlayLunarLanderVid(lunarLander: LunarLanderClass):
    lunarLander.env.reset()
    utils.play_video(lunarLander.env, lunarLander.q_network)

def SaveLunarLanderVids(NumVidsToSave, lunarLander: LunarLanderClass):
    lunarLander.env.reset()
    
    folder_path = "./videos"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filename = folder_path + "/lunar_lander"

    utils.create_videos(filename, lunarLander.env, lunarLander.q_network, NumVidsToSave)