import glob
import numpy as np
from torch.utils.data import Dataset


class Samples(Dataset):
    def __init__(self, data_dir):
        """
        :param data_dir: 数据集父目录
        :return:
        """
        self.allDataPath = []
        for dir in data_dir:
            self.allDataPath += glob.glob(dir + '/*.npy')
        self.allLabels = []
        for item in self.allDataPath:
            self.allLabels.append(int(item.split('_')[-3]))

    def __getitem__(self, index):
        """
        获取索引对应位置的一条数据
        :param index:
        :return:
        """
        data = np.load(self.allDataPath[index])  # 每一条数据的内容
        label = self.allLabels[index]  # 每一条数据的标签
        return data, label

    def __len__(self):
        """
        返回数据的总数量
        :return:
        """
        return len(self.allLabels)
