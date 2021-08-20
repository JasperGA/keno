import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # INSERT CODE HERE
        self.hid1 = nn.Linear(4*80, 10000)
        self.hid2 = nn.Linear(10000, 80)

        self.out = nn.Linear(80, 1)


    def forward(self, input):
        #output = 0*input[:,0] # CHANGE CODE HERE
        input = input.view(-1,4*80)
        self.x1 = torch.tanh(self.hid1(input))
        
        self.x2 = torch.tanh(self.hid2(self.x1))
        output = torch.sigmoid(self.out(self.x2))

        
        return output