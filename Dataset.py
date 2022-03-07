import torch
from torch.utils import data
import numpy as np
import random


class CellToDrugDataset(data.Dataset):
    def __init__(self, datas, datas_labels, datas_indexs, shuffle_index, train=True):
        # 数据集连同标签 一起打乱顺序
     
        all_index = range(len(datas))
        train_index = list(set(all_index).difference(set(shuffle_index)))
        if train:
           
            self.datas = datas[train_index]
            self.datas_labels = datas_labels[train_index]

            self.datas_indexs = datas_indexs[train_index]
        else:
            self.datas = datas[shuffle_index]
            self.datas_labels = datas_labels[shuffle_index]

            self.datas_indexs = datas_indexs[shuffle_index]

    def __getitem__(self, item):
        data = self.datas[item]
        data = (data - data.mean()) / data.std()
        label = self.datas_labels[item]
        data_index = self.datas_indexs[item]
        return data, label, data_index

    def __len__(self):
        return len(self.datas)

