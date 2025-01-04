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
from activation import ReLU, Softmax

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

# print(X_train.shape)

class Model(Module):
    def _init_(self):
        super()._init_()
        self.linear1 = Linear(28*28, 20)
        self.relu= ReLU()
        self.linear2 = Linear(20, 10)
        self.softmax=Softmax()
        
    def forward(self, x):
        x = self.linear1(x)    
        x = self.relu(x)       
        x = self.linear2(x)    
        x = self.softmax(x) 
        return x
    
model = Model()
print(model)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = CrossEntropyLoss()

for _ in range(501):
    x=Tensor(X_train, requires_grad=False)
    y=Tensor(Y_train, requires_grad=False)
    #print(y.data.shape)
    optimizer.zero_grad()
    y_hat = model(x)
    loss = loss_fn(y, y_hat)
    loss.backward()
    optimizer.step()
    # break
    if _ % 100 == 0:
        print(f'iteration: {_}')    
        print(f'Loss: {loss.data}')
        predictions = np.argmax(y_hat.data, axis=0)
        accuracy = np.sum(predictions == y.data) / y.data.size
        print(predictions, y.data)
        print(f'Accuracy: {accuracy * 100:.2f}%')
