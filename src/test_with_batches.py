import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from module import Module
from linear import Linear
from optimizer import SGD
from loss import CrossEntropyLoss, MSE
# from dataset import MNIST
# from dataloader import DataLoader
# from transformer import get_transform
from tensor import Tensor

# Load the dataset
data = pd.read_csv('../tests/data/MNIST.csv')

# Convert to numpy array for easy manipulation
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # Shuffle before splitting into dev and training sets

# Split into development and training sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255  # Normalize

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255  # Normalize
_, m_train = X_train.shape

# Define the Model
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

# Initialize the model, optimizer, and loss function
model = Model()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = CrossEntropyLoss()

# Batch size
batch_size = 32

# Training loop
for epoch in range(1):
    # Shuffle the training data at the start of each epoch
    permutation = np.random.permutation(m_train)
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[permutation]
    
    # Loop over the data in batches of size batch_size
    for i in range(0, m_train, batch_size):
        # Get the current batch
        X_batch = X_train_shuffled[:, i:i+batch_size]
        Y_batch = Y_train_shuffled[i:i+batch_size]
        
        # Convert to Tensor
        x = Tensor(X_batch, requires_grad=False)
        y = Tensor(Y_batch, requires_grad=False)

        print(f'x shape: {x.shape}, y shape: {y.shape} in batch {i//batch_size}')
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_hat = model(x)
        
        # Compute loss
        loss = loss_fn(y, y_hat)
        
        # Backpropagate
        loss.backward()
        
        # Update parameters
        optimizer.step()
    
    # Print progress every 100 epochs
    # if epoch % 100 == 0:
    print(f'Epoch: {epoch}')
    print(f'Loss: {loss.data}')
    
    # Compute accuracy on the training set
    predictions = np.argmax(y_hat.data, axis=0)
    accuracy = np.sum(predictions == y.data) / y.data.size
    print(f'Accuracy: {accuracy * 100:.2f}%')


correct = 0
total = 0
predictions_list = []

x=Tensor(X_dev, requires_grad=False)
y=Tensor(Y_dev, requires_grad=False)
    
y_hat = model(x)
predictions = np.argmax(y_hat.data, 0)
print(predictions)
correct = np.sum(predictions == y.data)
total = y.data.size
print(total)

# # # for pred, true in zip(predictions, y):
# # #     print(f'Prediction: {pred}, True Label: {true}')

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')