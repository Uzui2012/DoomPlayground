from collections import deque, namedtuple
from matplotlib import pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
def optimize_model(replay_mem, policy, target, optimiser, batch_size, gamma, device)-> float:
    if len(replay_mem.memory) < batch_size:
        return
    
    transitions = replay_mem.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.tensor(batch.reward).cuda()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    #print(next_state_values.is_cuda )
    with torch.no_grad():
        next_state_values[non_final_mask] = target(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    output_loss = loss.item()
    # Optimize the model
    optimiser.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimiser.step()
    return output_loss

def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('(Duration) Result')
    else:
        plt.clf()
        plt.title('(Duration) Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def plot_avg_scores(episode_scores, show_result=False):
    plt.figure(2)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('(Avg Score) Result')
    else:
        plt.clf()
        plt.title('(Avg Score) Training...')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')
    plt.plot(scores_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def plot_running_avg_scores(episode_scores, show_result=False):
    plt.figure(2)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('(Running Avg Score) Result')
    else:
        plt.clf()
        plt.title('(Running Avg Score) Training...')
    plt.xlabel('Episode')
    plt.ylabel('Running Avg Score')
    plt.plot(scores_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def plot_loss(loss_aggr, show_result=False):
    plt.figure(3)
    loss_aggr_t = torch.tensor(loss_aggr, dtype=torch.float)
    if show_result:
        plt.title('(Loss) Result')
    else:
        plt.clf()
        plt.title('(Loss) Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(loss_aggr_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    
