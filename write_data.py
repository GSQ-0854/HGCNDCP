import torch
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from utils import *
from config import args
from models import GCN, DNN, Net
# from build_datas import build_datas
device = torch.device('cpu')

def build_datas1():
    """构建数据样本"""
    labels = scio.loadmat('datas/CL_drug_triple.mat')['CL_drug_triple']  # 邻接矩阵
    adj = labels.copy()
   

    cellsim = scio.loadmat('datas/cellsim.mat')['cellsim']  # 细胞系相似性特征
    drugsim = scio.loadmat('datas/drugsim2.mat')['drugsim2']  # 药物相似性特征
    # 正样本邻接矩阵
    adj = np.vstack((np.hstack((np.zeros(shape=(962, 962), dtype=int), adj)),
                     np.hstack((adj.transpose(), np.zeros(shape=(183, 183), dtype=int)))))
    # 特征矩阵
    features = np.vstack((np.hstack((cellsim, np.zeros(shape=(962, 183), dtype=int))),
                          np.hstack((np.zeros(shape=(183, 962), dtype=int), drugsim))))
    size_cellsim = cellsim.shape  # (962,962)
    size_drugsim = drugsim.shape  # (183,183)
    adj = preprocess_adj(adj)
    features = normalize_features(features)
    return adj, features, size_cellsim, size_drugsim, labels

adj, features, size_cellsim, size_drugsim, labels = build_datas1()
print("adj type:", type(adj))
print(adj)
print("feature:", features)
print(labels.shape)
labels = torch.from_numpy(labels).to(device)
adj = torch.FloatTensor(adj).to(device)
features = torch.FloatTensor(features).to(device)

net = GCN(size_cellsim, size_drugsim, args.hidden, args.num_features_nonzero)
net.to(device)

optimizer1 = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
criterion1 = torch.nn.MSELoss()
for epoch in range(4000):
    cell_to_latent1, drug_to_latent1 = net([adj, features])
    outs = torch.mm(cell_to_latent1, drug_to_latent1.T)
    loss1 = criterion1(outs, labels.float())
    loss1 = loss1 + args.weight_decay * net.l2_loss()
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()
    print(epoch, loss1.item())

cell_to_latent = cell_to_latent1.detach().numpy()
drug_to_latent = drug_to_latent1.detach().numpy()
np.savetxt('datas/cell_to_latent_datas.txt', cell_to_latent)
np.savetxt('datas/drug_to_latent_datas.txt', drug_to_latent)
