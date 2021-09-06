import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, x_dim, u_dim):
        super(LSTM, self).__init__()

        