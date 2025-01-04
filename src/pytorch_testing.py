import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the model similar to your custom implementation
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(28*28, 20)  # First layer (784 -> 20)
        self.relu = nn.ReLU()  # ReLU activation
        self.linear2 = nn.Linear(20, 10)  # Second layer (20 -> 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)  # LogSoftmax for output layer

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.logsoftmax(x)

# Transformations for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss in PyTorch
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop (2 epochs)
for epoch in range(2):  # 2 epochs
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), -1).to(device)  # Flatten the input image (batch_size, 784)
        y = y.to(device)

        print(f'Batch {i+1}/{len(train_loader)}')
        print('Input shape:', x.shape)
        print('Target shape:', y.shape)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x)
        
        # Compute the loss
        loss = loss_fn(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        # Track running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    # Print statistics after each epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = (correct / total) * 100
    print(f"Epoch [{epoch+1}/2], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# After training, you can evaluate the model on test data if needed.