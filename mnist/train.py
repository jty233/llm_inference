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
hidden_size = 512  # number of nodes at hidden layer
num_classes = 10   # number of output classes discrete range [0,9]
num_epochs  = 30   # number of times which the entire dataset is passed throughout the model
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


def train(model_path):
    train_data = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_data  = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    assert len(train_data) == 60000
    assert len(test_data)  == 10000

    train_gen = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_gen  = torch.utils.data.DataLoader(dataset=test_data,  batch_size=batch_size, shuffle=False)

    net = Net(input_size, hidden_size, num_classes)

    if torch.cuda.is_available():
        net.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    t_start = time()
    for epoch in range(num_epochs):
        loss_history = []
        ncorrect = 0

        for i, (images, labels) in enumerate(train_gen):
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)

            loss_history.append(loss.cpu().data)
            _, predictions = torch.max(outputs, 1)
            ncorrect += (predictions == labels).sum()

            loss.backward()
            optimizer.step()

            if (i + 1)*batch_size % 10000 == 0:
                loss_mean = np.mean(loss_history)
                accuracy = ncorrect / ((i + 1) * batch_size)
                print(
                    f"Epoch [{epoch+1:02d}/{num_epochs}], "
                    f"Step [{(i+1)*batch_size:05d}/{len(train_data)}], "
                    f"Loss: {loss_mean:.4f}, Accuracy: {100*accuracy:.2f}%")
    print()
    print(f"Training took {time()-t_start:.2f}s")

    # Evaluate
    loss_history = []
    ncorrect = 0

    for i, (images, labels) in enumerate(test_gen):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        loss_history.append(loss.cpu().data)
        _, predictions = torch.max(outputs, 1)
        ncorrect += (predictions == labels).sum().cpu().numpy()

    loss_mean            = np.mean(loss_history)
    loss_uncertainty     = np.std(loss_history) / np.sqrt(len(loss_history) - 1)
    accuracy_mean        = ncorrect / (len(test_gen) * batch_size)
    accuracy_uncertainty = np.sqrt(accuracy_mean * (1.0 - accuracy_mean) / (len(test_gen) * batch_size))
    print()
    print(f"Test loss: {loss_mean:.6f}+-{loss_uncertainty:.6f}, Test accuracy: {100*accuracy_mean:.2f}+-{100*accuracy_uncertainty:.2f}%")

    # 获取模型权重
    state_dict = net.state_dict()

    # 保存为 .safetensors
    save_file(state_dict, "mnist.safetensors")


if __name__ == '__main__':
    train("mnist-fc")