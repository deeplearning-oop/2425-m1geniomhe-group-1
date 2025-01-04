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

from new_dataloader import DataLoader
import numpy as np
import pandas as pd

from new_dataset import MNIST

train_data = MNIST(root='data/', train=True, download=True)

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

res_x=None;result_y=None
for x, y in train_loader:
    x = x.flatten_batch()
    x=x.data
    if res_x is None:
        res_x=x
        print(res_x.shape)
    else:
        res_x = np.hstack((res_x, x))  # Stack vertically along rows
    y=y.data
    if result_y is not None:
        result_y = np.concatenate((result_y, y))
    else:
        result_y=y
    print(f'x shape:{x.shape}, resultx {res_x.shape}')
    print(f'y shape: {y.shape}, result y {result_y.shape}')

result=np.vstack((res_x, result_y.reshape(1,-1)))
rt=result.T

# Separate the last column as the index
index_labels = rt[:, -1]  # Last column for index
data = rt[:, :-1]  # All columns except the last one for data

# Create column labels
column_labels = [f'pixel{i}' for i in range(1, data.shape[1] + 1)]

# Create DataFrame
df = pd.DataFrame(data, columns=column_labels)
df.index = index_labels  # Set the last column as the index
df.index.name = 'label'  # Name the index column if needed

print('<> data processed from dataloaders to dataframe successfully')

# data = pd.read_csv('output.csv')

data = np.array(df)
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

for _ in range(501):
    x=Tensor(X_train, requires_grad=False)
    y=Tensor(Y_train, requires_grad=False)
    print(y.data.shape)
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