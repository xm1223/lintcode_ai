import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Neuralnetwork(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = torch.nn.Linear(in_dim, n_hidden_1)
        self.layer2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = torch.nn.Linear(n_hidden_2, out_dim)
 
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
