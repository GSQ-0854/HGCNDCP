import torch
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import random
from pandas import DataFrame
from utils import *
from config import args
from models import GCN, DNN, DNN_Con, Net
from Train import train
from Dataset import CellToDrugDataset
from read_data import read_data
from torchvision.models.resnet import resnet18
import sys, os
from tensorboardX import SummaryWriter
# from Train import build_datas

device = torch.device('cpu')
adj, features, size_cellsim, size_drugsim, labels, labels_mask = build_datas()
print("adj type:", type(adj))
# print(adj)
# print("feature:", features)
print(labels.shape)

labels = torch.from_numpy(labels)
labels_mask = torch.from_numpy(labels_mask)
# adj = torch.FloatTensor(adj)
# features = torch.FloatTensor(features)
adj = torch.tensor(adj, dtype=torch.float32, requires_grad=True)
features = torch.tensor(features, dtype=torch.float32, requires_grad=True)
hidden = [5, 25, 50, 75, 100]
extractor_loss = {}
for i in range(5):
    net = GCN(size_cellsim, size_drugsim, hidden[i], args.num_features_nonzero)
    net.to(device)
    # net.to(device)
    # cell_to_latent, drug_to_latent, new_output = net([adj, features])

    # print("cell_to_latent")
    # print(cell_to_latent)
    # print("cell_to_latent shape:", cell_to_latent.shape)
    #
    # print("drug_to_latent")
    # print(drug_to_latent)
    # print("drug_to_latent shape:", drug_to_latent.shape)
    #
    # print("new_output")
    # print(new_output)
    # print("new_output shape", new_output.shape)
    #
    # print("label")
    # print(labels)
    # print("labels shape", labels.shape)
    criterion = nn.BCEWithLogitsLoss()
    # m = nn.Sigmoid()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    net.train()
    sub_loss = []
    for epoch in range(2000):
        cell_to_latent, drug_to_latent, new_output = net([adj, features])
        # new_output = torch.tensor(new_output, dtype=torch.float32, requires_grad=True)
        labels = torch.tensor(labels, dtype=torch.float32, requires_grad=False)
        loss = criterion(new_output.reshape(-1)[labels_mask], labels.reshape(-1)[labels_mask])
        # loss = masked_loss(new_output, labels, labels_mask) hais
        # loss += args.weight_decay * net.l2_loss()

        optimizer.zero_grad()
        loss.backward()
        # loss = loss.requires_grad_()
        # for name, parms in net.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '-->grad_value:',parms.grad)
        optimizer.step()
        print("epoch :{}, train loss:{:.14f}".format(epoch, loss.item()))
        sub_loss.append(loss.item())

    extractor_loss[i] = sub_loss

    # net.eval()
    out = net([adj, features])
    print(out[2][0])
    print(labels[0])
    # for i in range(10):
    #     for j in range(labels.shape[1]):
    #         # if labels[i, j] == -1:
    #         #     # labels_mask[i, j] = 0
    #         #     labels[i, j] = 0  # 去负样本 只对正样本进行归一化操作
    #         print(out[2][i, :])
    cell_to_latent = out[0]
    drug_to_latent = out[1]
    np.save('extractor_result_data/cell_to_latent'+str(hidden[i]), cell_to_latent.data.cpu().numpy())
    np.save('extractor_result_data/drug_to_latent' + str(hidden[i]), drug_to_latent.data.cpu().numpy())
#
# datas = []
# datas_labels = []
# datas_indexs = []  # 为样本对增加相对应的索引
# pred_score = []   # 对最后的预测分数进行存储
# num_P = 0
#
# for i in tqdm(range(labels.shape[0])):
#     for j in range(labels.shape[1]):
#         if labels[i, j] == 1:
#             data = torch.hstack((cell_to_latent[i], drug_to_latent[j]))   # 向量拼接
#             label = 1
#             datas_indexs.append([i, j])  # 为样本对增加相对应的索引
#             datas.append(data)
#             datas_labels.append(label)
#         elif labels[i, j] == -1:  # 添加负样本
#             # num_P += 1
#             # if num_P % 2 == 0:
#             #     data = torch.hstack((cell_to_latent[i], drug_to_latent[j]))
#             #     label = 0
#             #     # data = data.reshape(16, 16)
#             #     datas.append(data)
#             #     datas_labels.append(label)
#             data = torch.hstack((cell_to_latent[i], drug_to_latent[j]))
#             label = 0
#             datas_indexs.append([i, j])    # 为样本对增加相对应的索引
#             datas.append(data)
#             datas_labels.append(label)
#         else:
#             pass
#
# datas = np.array(datas)
# datas_indexs = np.array(datas_indexs)
# datas_labels = np.array(datas_labels)
# # datas = datas.detach().numpy()
# # datas_indexs.detach().numpy()
# # datas_labels.detach().numpy()
# T = 10
# cv_num = 5
# shuffle_index = [i for i in range(len(datas))]
# random.shuffle(shuffle_index)
#
#
# for t in range(1):
#     order = div_list(shuffle_index, cv_num)     # 分5批次
#     for cv in range(cv_num):
#         print("cross_validation", '%01d' % cv)
#         index = order[cv]
#         train(index, t, cv)
print(cell_to_latent)
plt.title("extractor_loss")
plt.ylim(ymax=1, ymin=0)
plt.plot(np.arange(len(extractor_loss[0])), extractor_loss[0],
         label='latent factor is 5 (Best:{0:0.5f})'.format(min(extractor_loss[0])),
         color='aqua', linestyle='-', linewidth=1)
plt.plot(np.arange(len(extractor_loss[1])), extractor_loss[1],
         label='latent factor is 25 (Best:{0:0.5f})'.format(min(extractor_loss[1])),
         color='deeppink', linewidth=1)
plt.plot(np.arange(len(extractor_loss[2])), extractor_loss[2],
         label='latent factor is 50 (Best:{0:0.5f})'.format(min(extractor_loss[2])),
         color='darkorange', linewidth=1)
plt.plot(np.arange(len(extractor_loss[3])), extractor_loss[3],
         label='latent factor is 75 (Best:{0:0.5f})'.format(min(extractor_loss[3])),
         color='cornflowerblue', linewidth=1)
plt.plot(np.arange(len(extractor_loss[4])), extractor_loss[4],
         label='latent factor is 100 (Best:{0:0.5f})'.format(min(extractor_loss[4])),
         color='red', linewidth=1)
# plt.legend(['latent factor is 5 (Best:{0:0.5f})'.format(min(extractor_loss))], loc='upper right')
plt.legend(loc='upper right')
plt.show()
