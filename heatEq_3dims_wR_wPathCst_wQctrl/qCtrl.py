import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Ctrl(nn.Module):
    def __init__(self, x_dim, u_dim=2):
        super(Ctrl, self).__init__()

        self.input = nn.Linear(x_dim, (x_dim + u_dim) // 2)
        self.h1 = nn.Linear((x_dim + u_dim) // 2, (x_dim + u_dim) // 4)
        self.h2 = nn.Linear((x_dim + u_dim) // 4, (x_dim + u_dim) // 4)
        self.h3 = nn.Linear((x_dim + u_dim) // 4, u_dim)

        nn.init.kaiming_uniform_(self.input.weight)
        nn.init.kaiming_uniform_(self.h1.weight)
        nn.init.kaiming_uniform_(self.h2.weight)
        nn.init.kaiming_uniform_(self.h3.weight)

    def forward(self, x_in):
        x_h1 = F.leaky_relu(self.input(x_in))
        x_h2 = F.leaky_relu(self.h1(x_h1))
        x_h3 = F.leaky_relu(self.h2(x_h2))
        u_opt = (self.h3(x_h3))

        return u_opt


