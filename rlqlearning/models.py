import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, w, h, c, num_actions, conv_shape=[16, 32, 32], lin_shape=[512, 256]):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, conv_shape[0], kernel_size=3, stride=2) #16 32 32
        self.conv2 = nn.Conv2d(conv_shape[0], conv_shape[1], kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(conv_shape[1], conv_shape[2], kernel_size=3, stride=2)
        
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * conv_shape[2]
        
        self.fc1 = nn.Linear(linear_input_size, lin_shape[0]) #512 256
        self.fc2 = nn.Linear(lin_shape[0], lin_shape[1])
        self.fc3 = nn.Linear(lin_shape[1], num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
