import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import scale
from label_mapping import label_mapping


class ILDatasetSplit(Dataset):
    """Read data from the original dataset for feature extraction"""#从原始数据集读取数据进行特征提取

    def __init__(self, dataroot):#构造函数生成对象时，自动调用
        super(ILDatasetSplit, self).__init__()

        self.data = pd.read_csv(dataroot).to_numpy()
        self.feats = self.data[:, :520]
        self.label = label_mapping(self.data[:, 522:524])
        # self.scale = self.data[:, :520].max()
        # self.data = self.data / self.scale
        self.feats = scale(self.feats)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.feats[idx]
        label = self.label[idx]
        feature = torch.tensor(feature, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return feature, label


class ILDataset(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data):
        super(ILDataset, self).__init__()
        self.data = data
        self.feats = self.data[:, :520]
        self.label = label_mapping(self.data[:, 522:524])
        # self.scale = self.data[:, :520].max()
        # self.data = self.data / self.scale
        self.feats = scale(self.feats)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.feats[idx]
        label = self.label[idx]
        feature = torch.tensor(feature, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return feature, label


if __name__ == '__main__':
    # TestDataset = ILDataset('trainingData.csv')
    # TestDataset = ILDataset('validationData.csv')
    # TestDataloader = DataLoader(TestDataset, shuffle=True, batch_size=10)
    # for i, (feature, label) in enumerate(TestDataloader):
    #     print(feature.shape)
    #     print(label.shape)

    data1 = pd.read_csv('trainingData.csv').to_numpy()
    data2 = pd.read_csv('validationData.csv').to_numpy()
    data = np.concatenate([data1, data2])
    TestDataset = ILDataset(data)
    TestDataloader = DataLoader(TestDataset, shuffle=True, batch_size=10)
    for i, (feature, label) in enumerate(TestDataloader):
        print(feature.shape)
        print(label.shape)

