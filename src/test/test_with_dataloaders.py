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

from tensor import Tensor
import numpy as np

from dataset import MNIST
from dataloader import DataLoader

# -- using our implemented dataset module
train_data = MNIST(root='data/', train=True, download=True)
test_data = MNIST(root='data/', train=False, download=True)

# -- using our implemented dataloader module
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

# -- model definition
class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(20, 28*28)
        self.linear2 = Linear(10, 20)

    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x.softmax()
    
model = Model()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = CrossEntropyLoss()

# -- training

for epoch in range(5):
    for batch_no,(x, y) in enumerate(train_loader):
        # (32, 1, 28, 28)
        x=x.flatten_batch() # (784, 32)
        x=x/255
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