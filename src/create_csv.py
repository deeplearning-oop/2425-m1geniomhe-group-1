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

# Save to CSV
df.to_csv('../tests/MNIST_wannabe.csv')