# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split


# df_X=pd.read_csv('../data/data.csv')  
# df_y=pd.read_csv('../data/labels.csv')

# X_train, y_train, X_test, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
# print('-- data loaded')


# # Define the neural network class
# class GeneExpressionANN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(GeneExpressionANN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# # Function to calculate accuracy
# def calculate_accuracy(y_pred, y_true):
#     _, predicted = torch.max(y_pred, 1)
#     correct = (predicted == y_true).sum().item()
#     return correct / len(y_true)

# # Hyperparameters
# input_size = X_train.shape[1]  # Number of features
# hidden_size = 64              # Number of neurons in the hidden layer
# output_size = len(torch.unique(torch.tensor(y_train)))  # Number of classes
# learning_rate = 0.01
# num_epochs = 50
# batch_size = 32

# # Convert data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# # Create DataLoader for batching
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print('-- dataset object created')

# # Initialize the model, loss function, and optimizer
# model = GeneExpressionANN(input_size, hidden_size, output_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# # Store accuracies
# train_accuracies = []
# test_accuracies = []

# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0

#     for X_batch, y_batch in train_loader:
#         # Forward pass
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Accumulate loss and accuracy
#         total_loss += loss.item()
#         correct += (outputs.argmax(1) == y_batch).sum().item()
#         total += y_batch.size(0)

#     train_accuracy = correct / total
#     train_accuracies.append(train_accuracy)

#     # Evaluate on test data
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for X_batch, y_batch in test_loader:
#             outputs = model(X_batch)
#             correct += (outputs.argmax(1) == y_batch).sum().item()
#             total += y_batch.size(0)

#     test_accuracy = correct / total
#     test_accuracies.append(test_accuracy)

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

# # Save accuracies to files
# np.savetxt("res/gene_expression/torch_train_accuracies.csv", train_accuracies, delimiter=",")
# np.savetxt("res/gene_expression/test_accuracies.csv", test_accuracies, delimiter=",")

# # Save the trained model
# torch.save(model.state_dict(), "res/gene_expression/model.pth")

# print("-- training complete: model and accuracies saved.")
