import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from module import Module
from linear import Linear
from optimizer import SGD
from loss import CrossEntropyLoss, MSE
from activation import ReLU, Softmax
from tensor import Tensor
import numpy as np

from dataset import MNIST
from dataloader import DataLoader
from transforms import Compose, ToTensor, Normalize, Standardize

# -- using our implemented dataset module
transformation=Compose([ToTensor(), Standardize()])
train_data = MNIST(root='data/', train=True, download=True,transform=transformation)
test_data = MNIST(root='data/', train=False, download=True,transform=transformation)

# -- using our implemented dataloader module
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

# -- model definition
class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28*28,20)
        self.relu=ReLU()
        self.linear2 = Linear(20, 10)
        self.softmax=Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.softmax(x)
    
model = Model()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = CrossEntropyLoss()

# -- training

# List to store accuracies from each run
accuracies = []

# Run the experiment 10 times
for run in range(10):
    print(f"Run {run + 1} / 10:")
    
    # -- training loop
    for epoch in range(1):  # You can adjust the number of epochs as needed
        for batch_no, (x, y) in enumerate(train_loader):
            # Flatten the batch (32, 1, 28, 28) to (784, 32)
            x = x.flatten_batch()  # (784, 32)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y, y_hat)
            loss.backward()
            optimizer.step()
    
    # -- testing
    correct = 0
    total = 0

    for batch_no, (x, y) in enumerate(test_loader):
        x = x.flatten_batch()
        y_hat = model(x)
        predictions = np.argmax(y_hat, axis=0)
        correct += np.sum(predictions == y)
        total += y.data.size
    
    accuracy = correct / total * 100
    accuracies.append(accuracy)
    
    print(f'Accuracy for run {run + 1}: {accuracy:.2f}%')
    print('------------------')

# Optionally, you can calculate and print the average accuracy across all runs
average_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy over 10 runs: {average_accuracy:.2f}%')


# compare to pytorch implementation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = datasets.MNIST('data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(28*28, 20), nn.ReLU(), nn.Linear(20, 10), nn.Softmax(dim=1))
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

# List to store accuracies from each run
pytorch_accuracies = []

# Run the experiment 10 times
for run in range(10):
    print(f"Run {run + 1} / 10:")
    
    # -- training loop
    for epoch in range(1):  # You can adjust the number of epochs as needed
        for batch_no, (x, y) in enumerate(trainloader):
            x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
    
    # -- testing
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation during testing
        for batch_no, (x, y) in enumerate(testloader):
            x = x.view(x.shape[0], -1)
            y_hat = model(x)
            predictions = torch.argmax(y_hat, dim=1)
            correct += torch.sum(predictions == y)
            total += y.size(0)
    
    accuracy = correct / total * 100
    pytorch_accuracies.append(accuracy)
    
    print(f'Pytorch Accuracy for run {run + 1}: {accuracy:.2f}%')
    print('------------------')

# Optionally, you can calculate and print the average accuracy across all runs
average_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy over 10 runs: {average_accuracy:.2f}%')

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), pytorch_accuracies, label='PyTorch Accuracy', marker='o', color='blue')
plt.plot(range(1, 11), accuracies, label='Custom Accuracy', marker='x', color='red')

plt.xlabel('Run Number')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of PyTorch and Custom Accuracy over 10 Runs')
plt.legend()
plt.grid(True)
plt.show()
        
    
