from ann.nn.module import Module
from ann.nn.linear import Linear
from ann.nn.optimizer import SGD
from ann.nn.loss import CrossEntropyLoss
from ann.nn.activation import ReLU, Softmax

from ann.extensions.dataset import MNIST
from ann.extensions.dataloader import DataLoader
from ann.extensions.transforms import Compose, ToTensor, Standardize

import numpy as np
import matplotlib.pyplot as plt

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

average_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy over 10 runs: {average_accuracy:.2f}%')

plt.plot(accuracies)