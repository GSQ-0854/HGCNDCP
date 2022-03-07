import torch
from torch import nn
from torch.nn import functional as F
from layers import *
from config import args

from torchvision.models.resnet import resnet18


class GCN(nn.Module):

    def __init__(self, cellsim_feature_shape, drugsim_feature_shape,
                 latent_factor_num, num_features_nonzero,):
        super(GCN, self).__init__()

        self.cell_feature_shape = cellsim_feature_shape
        self.drugsim_feature_shape = drugsim_feature_shape
        self.latent_factor_num = latent_factor_num
        self.num_features_nonzero = num_features_nonzero
        print('latent factor num:', latent_factor_num)
        print('num features nonzero:', num_features_nonzero)

        self.layers = nn.Sequential(GraphConvolution(self.cell_feature_shape,
                                                     self.drugsim_feature_shape,
                                                     self.latent_factor_num,
                                                     self.num_features_nonzero,
                                                     activation=F.leaky_relu),
                                    Decoder(self.cell_feature_shape,
                                            self.drugsim_feature_shape,
                                            self.latent_factor_num))

    def forward(self, inputs):

        x = self.layers[0](inputs)
        cell_to_latent, drug_to_latent, new_output = self.layers[1](x)
        return cell_to_latent, drug_to_latent, new_output

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()  # 依次累加损失值

        return loss


class DNN(nn.Module):
    """构建DNN网络,对正负样本进行预测"""
    def __init__(self, in_dim):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(in_dim, 128)
        # self.linear2 = nn.Linear(128, 450)
        # self.linear3 = nn.Linear(450, 256)
        self.linear4 = nn.Linear(128, 96)
        self.linear5 = nn.Linear(96, 64)
        self.linear6 = nn.Linear(64, 2)

        self.batchNorm1 = nn.BatchNorm1d(num_features=128)
      
        self.batchNorm4 = nn.BatchNorm1d(num_features=96)
        self.batchNorm5 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):

        x = self.linear1(x)  # 256
        x = self.batchNorm1(x)
        x = F.dropout(x, p=0.18)
        # x = F.leaky_relu(x)
        x = F.relu(x)

         x = self.linear4(x)
        x = self.batchNorm4(x)   # 64
        x = F.dropout(x, p=0.18)
        x = F.leaky_relu(x)
        # x = F.relu(x)

        x = self.linear5(x)
        x = self.batchNorm5(x)
        # x = F.dropout(x, p=0.15)
        # x = F.leaky_relu(x)  # 2
        x = F.relu(x)

        x = self.linear6(x)
        x = F.relu(x)
        # x = F.softmax(x, dim=1)
        return x

    def l2_loss(self):

        layer = self.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()  # 依次累加损失值

        return loss


class DNN_Con(nn.Module):
    def __init__(self):
        super(DNN_Con, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(576, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 3)
        )
     

    def forward(self, x):
        #
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def l2_loss(self):

        layer = self.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()  # 依次累加损失值

        return loss


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(512, 3)  # 加上一层参数修改好的全连接层

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


