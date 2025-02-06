import torch
from time import sleep, time
from vizdoom import gymnasium_wrapper
import gymnasium

use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
device = torch.device("cuda:0" if use_cuda else "cpu")
env = gymnasium.make("VizdoomHealthGatheringSupreme-v0", render_mode="human")

for i in range(5):
    done = False
    observation = env.reset()
    score = 0

    while not done:  
        action = env.action_space.sample()
        observation, reward, done, trunc, info = env.step(action)
        score += reward
        print(score)

        env.render()
        #sleep(1/30)

        if done:
            print(f"Episode Score: {score}")

