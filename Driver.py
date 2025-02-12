# %%
import torch
import matplotlib
import matplotlib.pyplot as plt
from time import sleep, time
from vizdoom import gymnasium_wrapper
import gymnasium
import numpy as np
import random
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
env = gymnasium.make("VizdoomBasic-v0", render_mode="human")

# %%
import rlqlearning.rlqlUtil as util
import rlqlearning.models as models

# %%
BATCH_SIZE = 128 # Transitions sampled from replay buffer
GAMMA = 0.99 # Discount factor
EPSILON_START = 0.9 # Start, end, and rate of decay of the epsilon value 
EPSILON_END = 0.05
EPSILON_DECAY = 1000
TAU = 0.005 # Target network update rate
LEARNING_RATE = 1e-4 # optimiser learning rate
num_actions = env.action_space.n
state, info = env.reset()
screen_w = len(state['screen'])
screen_h = len(state['screen'][0])
channels =  3
#print(screen_w)
#print(screen_h)
screen_state = torch.tensor(np.array(state['screen']), dtype=torch.float32, device=device).unsqueeze(0)

# %%
policy = models.DQN(screen_w, screen_h, channels, num_actions).to(device)
target = models.DQN(screen_w, screen_h, channels, num_actions).to(device)
target.load_state_dict(policy.state_dict())

optimiser = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = util.ReplayMemory(10000)

# %%
steps_done = 0
episode_durations = []

def select_action(observation):
    global steps_done
    sample = random.random()
    eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
        math.exp(-1. * steps_done / EPSILON_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy(observation).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# %%
if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 50

for i in range(num_episodes):
    done = False
    state, info = env.reset()
    state = torch.tensor(np.array(state['screen']), dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.transpose(state, 1, 3)
    print(state.shape)
    score = 0
    while not done:
        action = select_action(state)
        print(action.item())
        observation, reward, done, trunc, info = env.step(action.item())
        score += reward
        if done:
            next_state = None
            print(f"Episode Score: {score}")
        else:
            next_state = torch.tensor(observation['screen'], dtype=torch.float32, device=device).unsqueeze(0)
            next_state = torch.transpose(next_state, 1, 3)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        #env.render()
        #sleep(1/30)


