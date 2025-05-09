import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from safetensors.torch import save_file

import sys
import os
import json
from time import time

input_size  = 784  # img_size = (28,28) ---> 28*28=784 in total
hidden_size = 500  # number of nodes at hidden layer
num_classes = 10   # number of output classes discrete range [0,9]
num_epochs  = 10   # number of times which the entire dataset is passed throughout the model
batch_size  = 500  # the size of input data took for one iteration
lr          = 1e-3 # size of step


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':


    a = np.random.randn(1000000, 784).astype(np.float32)
    b = np.random.rand(784, 512).astype(np.float32)
    t_start = time()
    c = a @ b
    print(f"numpy took {time()-t_start:.3f}s")


    d = torch.randn((1000000, 784))
    e = torch.rand((784, 512))
    t_start = time()
    f = d @ e
    print(f"torch took {time()-t_start:.3f}s")


    train_data = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_data  = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    assert len(train_data) == 60000
    assert len(test_data)  == 10000

    test_gen  = torch.utils.data.DataLoader(dataset=test_data,  batch_size=10000, shuffle=False)

    net = Net(input_size, hidden_size, num_classes)


    t_start = time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_gen):
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
            outputs = net(images)

    print(f"test took {time()-t_start:.3f}s")