from torch.utils.data import DataLoader, Sampler
from torchvision import datasets,transforms
import torch

class mydataset(torch.utils.data.Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __getitem__(self, idx):

        return torch.tensor(self.input[idx]),torch.tensor(self.output[idx])

    def __len__(self):
        return len(self.input)