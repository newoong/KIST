import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary

class Projection(nn.Module):
    def __init__(self, c_in, hidden_dim, c_out):
        super().__init__()
        self.fc_1 = nn.Linear(c_in, hidden_dim)
        self.fc_2= nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, c_out)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.shape[0],-1)
            
        x = self.relu(self.fc_1(inputs))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        
        return x
