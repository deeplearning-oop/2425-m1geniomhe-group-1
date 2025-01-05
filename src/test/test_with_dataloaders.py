import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
optimizer = SGD(model.parameters(), lr=0.1)
loss_fn = CrossEntropyLoss()

# -- training

for epoch in range(1):
    for batch_no,(x, y) in enumerate(train_loader):
        # (32, 1, 28, 28)
        x=x.flatten_batch() # (784, 32)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y, y_hat)
        loss.backward()
        optimizer.step()
        

    print(f'iteration: {epoch}')    
    print(f'Loss: {loss.data}') 
    predictions = np.argmax(y_hat.data, axis=0)
    accuracy = np.sum(predictions == y.data) / y.data.size
    print(predictions, y.data)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('------------------')

# for batch, (x,y) in enumerate(test_loader):
#     x=x.flatten_batch()
#     y_hat=model(x)
#     loss = loss_fn(y, y_hat)
    