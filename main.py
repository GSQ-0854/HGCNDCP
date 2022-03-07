import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import functional as F

import numpy as np
from utils import *
from config import args
from models import GCN
import sys, os
from train import train

sys.path.append(os.pardir)

seed = 100
np.random.seed(seed)
torch.random.manual_seed(seed)

labels = np.loadtxt('datas/adj_p.txt')
reorder = np.arange(labels.shape[0])
np.random.shuffle(reorder)

T = 10
cv_num = 5
for t in range(1):
    order = div_list(reorder.tolist(), cv_num)
    for i in range(1):
        print("cross_validation:", '%01d' % i)
        test_arr = order[i]
        arr = list(set(reorder).difference(set(test_arr)))
        np.random.shuffle(arr)
        train_arr = arr
        A = train(train_arr, test_arr)
        print('test_arr length:', len(test_arr))
        # print(test_arr)
        print('train_arr length:', len(train_arr))
        # print(train_arr)

