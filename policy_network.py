"""
A 4x256x256 neural network for approximating pi(a | s)
@author yubaidi
"""
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
  def __init__(self, D_in, D_h1, D_h2, D_out):
    super(PolicyNetwork, self).__init__()
    self.fc1 = nn.Linear(D_in, D_h1)
    self.fc2 = nn.Linear(D_h1, D_h2)
    self.fc3 = nn.Linear(D_h2, D_out)

  def forward(self, x):
    h1 = F.relu(self.fc1(x))
    h2 = F.relu(self.fc2(h1))
    output = F.softmax(self.fc3(h2))
    return output



