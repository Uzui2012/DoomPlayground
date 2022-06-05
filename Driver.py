from logging import exception
from vizdoom import DoomGame
from vizdoom import gym_wrapper
import gym
from random import choice
from time import sleep, time

from DQN import DQN_agent

env = gym.make("VizdoomBasic-v0")
#print(f"OBSERVATION SPACE: {env.observation_space[0].n}")
#print(f"ACTION SPACE: {env.action_space[0]}")
dqn_agent = DQN_agent(3,1)

for i in range(1000):
    done = False
    observation = env.reset()
    print(observation)
    while not done:
        
        #action = dqn_agent.select_action(observation)
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()


