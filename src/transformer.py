import torchvision.transforms as transforms

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize images to have mean 0.5 and std 0.5
    ])