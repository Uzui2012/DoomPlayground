import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

###### PARAMS ######
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

num_eps = 50000

####################


def calc_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
                math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

def load_model():
    return torch.load("DoomDQN_save.pth")

def save_model(model):
    torch.save(model.state_dict(), "DoomDQN_save.pth")

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        if(self.position >= len(self.memory)):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)

class Model():
    def __init__(self, num_actions, num_frames):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_frames, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.tick = 0
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)


    def forward(self, observation):
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), 7 * 7 * 64))) # was 4* 4
        x = self.fc2(x)

        return x

class DQN_agent():
    def __init__(self, num_actions, num_frames):
        self.egreedy = 0.9
        self.egreedy_final = 0.01
        self.egreedy_decay = 150000
        self.replay_mem_size = 100000
        self.batch_size = 64
        self.learning_rate = 0.00025
        self.gamma = 0.99
        self.update_target_frequency = 10000
        self.clip_error = True
        self.double_dqn = True

        self.num_actions = num_actions
        self.num_frames = num_frames
        self.memory = ExperienceReplay(self.replay_mem_size)
        self.model = Model(num_actions, num_frames)
        
    
    def select_action(self, observation):
        random_for_egreedy = torch.rand(1)[0]        
        if random_for_egreedy > self.epsilon:
            with torch.no_grad():
                action_from_nn = self.model(observation)
                action = torch.max(action_from_nn,1)[1]
                action = action.item()
        else:
            action = random.randrange(0, self.num_actions)

    def optimize(self):
        if(len(self.memory) < self.batch_size):
            return
        state, action, new_state, reward, done = self.memory.sample(self.batch_size)
        
        state = torch.cat(state)

        new_state = torch.cat(new_state)

        reward = torch.Tensor(reward).to(device)
        action = torch.LongTensor(action).to(device)
        done = torch.Tensor(done).to(device)

        if self.double_dqn:
            new_state_indexes = self.model(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]  
            
            new_state_values = self.target_model(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_model(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]
        
        #Bellman Equ
        #Q[state, action] = reward + gamma*torch.max(Q[new_state])
        target_val = reward + (1 - done)*self.gamma * max_new_state_values

        predicted_val = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1) # Now we want to calculate gradient
        loss = self.criterion(predicted_val, target_val)
        self.optimizer.zero_grad()
        loss.backward()

        if(self.clip_error):
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.num_of_frames % self.update_target_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        #if self.num_of_frames % save_model_freq == 0:
        #    save_model(self.model)

        self.num_of_frames += 1