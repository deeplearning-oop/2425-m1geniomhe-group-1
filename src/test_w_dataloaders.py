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
import numpy as np

from new_dataset import MNIST
from new_dataloader import DataLoader



data = pd.read_csv('../tests/data/MNIST.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape

# for _ in range(501):
#     x=Tensor(X_train, requires_grad=False)
#     y=Tensor(Y_train, requires_grad=False)
#     print(x,x.shape)
#     print(y,y.shape)
#     break

train_data = MNIST(root='data/', train=True, download=True)
test_data = MNIST(root='data/', train=False, download=True)

# print(train_data)
# print(test_data)

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

for i,(x, y) in enumerate(train_loader):
    print(f'batch number: {i} | x shape: {x.shape}  |  y: {y}, type y: {type(y)}')
    if i==5:
        break

# print(X_train.shape)

##########################################

# class Model(Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = Linear(20, 28*28)
#         self.linear2 = Linear(10, 20)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = x.relu()
#         x = self.linear2(x)
#         return x.softmax()
    
# model = Model()
# optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
# loss_fn = CrossEntropyLoss()

# for _ in range(501):
#     x=Tensor(X_train, requires_grad=False)
#     y=Tensor(Y_train, requires_grad=False)
#     # print(y.data.shape)
#     optimizer.zero_grad()
#     y_hat = model(x)
#     loss = loss_fn(y, y_hat)
#     loss.backward()
#     optimizer.step()
#     # break
#     if _ % 100 == 0:
#         print(f'iteration: {_}')    
#         print(f'Loss: {loss.data}') 
#         predictions = np.argmax(y_hat.data, axis=0)
#         accuracy = np.sum(predictions == y.data) / y.data.size
#         print(predictions, y.data)
#         print(f'Accuracy: {accuracy * 100:.2f}%')
    

################################################################

# correct = 0
# total = 0
# predictions_list = []

# x=Tensor(X_dev, requires_grad=False)
# y=Tensor(Y_dev, requires_grad=False)
    
# y_hat = model(x)
# predictions = np.argmax(y_hat.data, 0)
# print(predictions)
# correct = np.sum(predictions == y.data)
# total = y.data.size
# print(total)

# # # for pred, true in zip(predictions, y):
# # #     print(f'Prediction: {pred}, True Label: {true}')

# accuracy = correct / total
# print(f'Accuracy: {accuracy * 100:.2f}%')

# # # Print frequency of each prediction
# # # unique, counts = np.unique(predictions_list, return_counts=True)
# # # print(f'Frequency of each prediction: {dict(zip(unique, counts))}')