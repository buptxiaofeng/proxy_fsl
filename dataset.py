import torch
import numpy
from torch.utils.data.dataset import Dataset
from data import MiniImagenet

class MiniDataset(Dataset):

    def __init__(self, data_type, num_way, num_shot, episodes):
        assert data_type == "train" or data_type == "test" or data_type == "val"

        self.episodes = episodes
        self.num_way = num_way
        self.num_shot = num_shot
        self.data_type = data_type
        self.data = MiniImagenet()

    def __getitem__(self):
        self.load_data(self.data_type, self.num_way, self.num_shot)

    def __len__(self):

        return self.episodes
