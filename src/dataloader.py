import numpy as np

class DataLoader:
    def __init__(self, dataset, batchsize=32, shuffle=True):
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batchsize):
            batch_indices = self.indices[i:i + self.batchsize]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield batch