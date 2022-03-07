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

print(labels.shape)

labels = torch.from_numpy(labels)
labels_mask = torch.from_numpy(labels_mask)

adj = torch.tensor(adj, dtype=torch.float32, requires_grad=True)
features = torch.tensor(features, dtype=torch.float32, requires_grad=True)
hidden = [5, 25, 50, 75, 100]
extractor_loss = {}
for i in range(5):
    net = GCN(size_cellsim, size_drugsim, hidden[i], args.num_features_nonzero)
    net.to(device)
  
    criterion = nn.BCEWithLogitsLoss()
 
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    net.train()
    sub_loss = []
    for epoch in range(2000):
        cell_to_latent, drug_to_latent, new_output = net([adj, features])
        
        labels = torch.tensor(labels, dtype=torch.float32, requires_grad=False)
        loss = criterion(new_output.reshape(-1)[labels_mask], labels.reshape(-1)[labels_mask])


        optimizer.zero_grad()
        loss.backward()
   
        optimizer.step()
        print("epoch :{}, train loss:{:.14f}".format(epoch, loss.item()))
        sub_loss.append(loss.item())

    extractor_loss[i] = sub_loss

    # net.eval()
    out = net([adj, features])
    print(out[2][0])
    print(labels[0])

    cell_to_latent = out[0]
    drug_to_latent = out[1]
    np.save('extractor_result_data/cell_to_latent'+str(hidden[i]), cell_to_latent.data.cpu().numpy())
    np.save('extractor_result_data/drug_to_latent' + str(hidden[i]), drug_to_latent.data.cpu().numpy())

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

plt.legend(loc='upper right')
plt.show()
