
import torch
import gym
import numpy
from time import sleep, time
from vizdoom import gymnasium_wrapper
import gymnasium

#from DQN import DQN_agent

use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
device = torch.device("cuda:0" if use_cuda else "cpu")
env = gymnasium.make("VizdoomHealthGatheringSupreme-v0", render_mode="human")
#print(f"OBSERVATION SPACE: {env.observation_space[0].n}")
#print(f"ACTION SPACE: {env.action_space[0]}")
#dqn_agent = DQN_agent(3,3)

def format_frame(observation):
    frame = observation.get("rgb")
    frame = torch.from_numpy(frame)
    frame = frame.float()
    frame = frame.unsqueeze(0)
    frame = frame.permute(0, 3, 1, 2)
    return frame


for i in range(5):
    done = False
    observation = env.reset()
    #frame = format_frame(observation)
    #game_vars = observation.get("gamevariables")
    
    score = 0


    while not done:  
        action = env.action_space.sample()
        observation, reward, done, trunc, info = env.step(action)
        #new_frame = format_frame(observation)
        #print(f"Action chosen: {action}, gained reward {reward}")
        score += reward
        #dqn_agent.memory.push(frame, action, new_frame, reward, done)
        #dqn_agent.optimize()
        print(score)
        #frame = new_frame

        env.render()
        #sleep(1/30)

        if done:
            print(f"Episode Score: {score}")

