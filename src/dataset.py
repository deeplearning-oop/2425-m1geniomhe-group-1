import torchvision.transforms as transforms
from torchvision.datasets import MNIST as TorchvisionMNIST

class MNIST:
    def __init__(self, root='./data', train=True, transform=None, download=True):
        self.dataset = TorchvisionMNIST(root=root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)