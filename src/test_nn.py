from module import Module
from linear import Linear
# from activation import ReLU, LogSoftmax
from optimizer import SGD
from loss import CrossEntropyLoss, MSE
from dataset import MNIST
from dataloader import DataLoader
from transformer import get_transform
from tensor import Tensor
import numpy as np


class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(20,28*28)
        self.linear2 = Linear(10,20)

    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x.softmax()

model = Model()
print(model)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = CrossEntropyLoss()
transform = get_transform()
dataset = MNIST(transform=transform)
dataloader = DataLoader(dataset, batchsize=32, shuffle=True)

# # # Training loop
for epoch in range(3):
    print(f'epoch: {epoch+1} started')
    for batch in dataloader:
        x, y = zip(*batch)
        x = Tensor(np.array([img.numpy().flatten() for img in x]).T / 255, requires_grad=False) # Normalize input
        y = Tensor(np.array(y), requires_grad=False)  
        # y = Tensor(np.eye(10)[np.array(y)], requires_grad=False)  # One-hot encode labels

        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y, y_hat)
        loss.backward()
        optimizer.step()
    
       
import pickle

# Save the model parameters (weights and biases)
def save_model(model, filename="model.pkl"):
    # Save model weights
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Call this function after training
# save_model(model)


# Performance assessment
correct = 0
total = 0
predictions_list = []
for batch in dataloader:
    x, y = zip(*batch)
    x = Tensor(np.array([img.numpy().flatten() for img in x]).T / 255, requires_grad=False)  # Normalize input
    y = np.array(y)
    
    y_hat = model(x)
    predictions = np.argmax(y_hat.data, axis=0)
    correct += (predictions == y).sum()
    total += y.size
    predictions_list.extend(predictions)

    # for pred, true in zip(predictions, y):
    #     print(f'Prediction: {pred}, True Label: {true}')

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print frequency of each prediction
unique, counts = np.unique(predictions_list, return_counts=True)
print(f'Frequency of each prediction: {dict(zip(unique, counts))}')