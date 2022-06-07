from vizdoom import gym_wrapper
import torch
import gym
import numpy
from time import sleep, time

from DQN import DQN_agent

env = gym.make("VizdoomBasic-v0")
#print(f"OBSERVATION SPACE: {env.observation_space[0].n}")
#print(f"ACTION SPACE: {env.action_space[0]}")
dqn_agent = DQN_agent(3,3)

def format_frame(observation):
    frame = observation.get("rgb")
    frame = torch.from_numpy(frame)
    frame = frame.float()
    frame = frame.unsqueeze(0)
    frame = frame.permute(0, 3, 1, 2)
    return frame

for i in range(20):
    done = False
    observation = env.reset()
    frame = format_frame(observation)
    game_vars = observation.get("gamevariables")
    
    score = 0


    while not done:  
        action = dqn_agent.select_action(frame)
        observation, reward, done, info = env.step(action)
        new_frame = format_frame(observation)
        #print(f"Action chosen: {action}, gained reward {reward}")
        score += reward
        #dqn_agent.memory.push(frame, action, new_frame, reward, done)
        #dqn_agent.optimize()

        frame = new_frame

        env.render()
        #sleep(1/40)

        if done:
            print(f"Episode Score: {score}")


