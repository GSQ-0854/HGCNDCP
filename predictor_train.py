import torch
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import random
from pandas import DataFrame
from utils import *
from config import args
from models import GCN, DNN, DNN_Con, Net
from Dataset import CellToDrugDataset
from read_data import read_data
from torchvision.models.resnet import resnet18
import sys, os
from tensorboardX import SummaryWriter
device = torch.device('cpu')

cell_to_latent100 = np.load('extractor_result_data/cell_to_latent100.npy')
drug_to_latent100 = np.load('extractor_result_data/drug_to_latent100.npy')

result = build_datas()
labels_mask = result[5]
labels = torch.from_numpy(result[4])
print("labels:", type(labels), labels.shape)
print("cell_to_latent_100:", type(cell_to_latent100), cell_to_latent100.shape)
print("drug_to_latent_100:", type(drug_to_latent100), drug_to_latent100.shape)

# adj = torch.FloatTensor(adj).to()
# labels = torch.from_numpy(labels).to(device)
# cell_to_latent100 *= 500
# drug_to_latent100 *= 500
cell_to_latent100 = torch.FloatTensor(cell_to_latent100)
drug_to_latent100 = torch.FloatTensor(drug_to_latent100)

data_sets = []
data_set_labels = []
data_set_indexs = []  # 为样本对增加相对应的索引
pred_score = []       # 对最后的预测分数进行存储
num_P = 0
negative_ratio = 5

# 向量拼接
for i in tqdm(range(labels.shape[0])):
    for j in range(labels.shape[1]):
        if labels[i, j] == 1:
            data_set = np.hstack((cell_to_latent100[i], drug_to_latent100[j]))  # 向量拼接
            label = 1
            data_sets.append(data_set)       # 添加拼接后的数据
            data_set_labels.append(label)    # 添加对应标签
            data_set_indexs.append([i, j])   # 添加对应的向量位置索引
        elif labels[i, j] == -1:
            num_P += 1
            if num_P < (16804 * negative_ratio):     # 按比例添加负样本
                data_set = np.hstack((cell_to_latent100[i], drug_to_latent100[j]))
                label = 0
                data_sets.append(data_set)
                data_set_labels.append(labels[i, j] + 1)
                data_set_indexs.append([i, j])

            # data_set = np.hstack((cell_to_latent100[i], drug_to_latent100[j]))
            # label = 0
            # data_sets.append(data_set)
            # data_set_labels.append(label)
            # data_set_indexs.append([i, j])
        else:
            pass


data_sets = np.array(data_sets)
data_set_labels = np.array(data_set_labels)
data_set_indexs = np.array(data_set_indexs)

print(type(data_sets), type(data_set_labels))
print(len(data_sets), len(data_set_labels))

Datas = torch.from_numpy(data_sets)
Labels = torch.from_numpy(data_set_labels).long()
Datas_Index = torch.from_numpy(data_set_indexs)
# 生成乱序的数据索引
shuffle_indexs = [i for i in range(len(data_sets))]
random.shuffle(shuffle_indexs)


def train(shuffle_index, name, t, cv):
    """模型训练"""
    # 构造训练模型的数据集
    train_dataset = CellToDrugDataset(Datas, Labels, Datas_Index, shuffle_index)
    test_dataset = CellToDrugDataset(Datas, Labels, Datas_Index, shuffle_index, train=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # # 单条数据展示
    # examples = enumerate(train_loader)
    # batch_id, (examples_data, examples_labels, examples_indexs) = next(examples)
    # print(batch_id, examples_data, examples_labels)
    # 定义DNN网络模型
    dnn = DNN(args.hidden * 2)
    dnn.to(device)
    print(dnn)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(dnn.parameters(), lr=args.learning_rate)   # 优化器
    predictor_loss = []
    predictor_acc = []
    predictor_eval_loss = []
    predictor_eval_acc = []

    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        dnn.train()  # 训练模式
        # # 动态修改学习率参数
        # if epoch % 5 == 0:
        #     optimizer.param_groups[0]['lr'] *= 0.1
        for data, data_label, data_index in train_loader:
            data = data.to(device)
            # data = data.clone().detach()
            # data_label = torch.as_tensor(data_label, dtype=torch.long).to(device)
            data_label = data_label.to(device)
            data_index = data_index.to(device)

            data = data.view(data.size(0), -1)
            out = dnn(data)
            loss = criterion(out, data_label)
            loss = loss + args.weight_decay * dnn.l2_loss()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # for name, parms in dnn.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '-->grad_value:',parms.grad)
            train_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == data_label).sum().item()
            acc = num_correct / data.shape[0]
            train_acc += acc

        predictor_loss.append(train_loss / len(train_loader))
        predictor_acc.append(train_acc / len(train_loader))

        eval_loss = 0
        eval_acc = 0
        score_list = []
        label_list = []
        index_list = []
        # 将模型改为预测模式
        dnn.eval()
        for data, data_label, data_index in test_loader:
            # data = data.to(device)
            data = data.to(device)
            data_label = data_label.to(device)
            data_index = data_index.to(device)

            data = data.view(data.size(0), -1)
            out = dnn(data)
            score_temp = out
            # print(out)
            loss = criterion(out, data_label)
            eval_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == data_label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc

            score_list.extend(score_temp.detach().numpy())
            label_list.extend(data_label.numpy())
            index_list.extend(data_index)

        predictor_eval_loss.append(eval_loss / len(test_loader))
        predictor_eval_acc.append(eval_acc / len(test_loader))
        print('epoch:{},Train Loss: {:.6f},Train Acc:{:.6f}, Test Loss:{:.6f},Test Acc: {:.6f}'
              .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                      eval_loss / len(test_loader), eval_acc / len(test_loader)))

    plt.subplot(2, 2, 1)
    plt.title("train_loss")
    plt.plot(np.arange(len(predictor_loss)), predictor_loss)
    plt.legend(['Train Loss (Best:{0:0.5f})'.format(min(predictor_loss))], loc='upper right')
    # plt.show()

    titles = ['Train Loss', 'Train Acc', 'Test Loss', 'Test Acc']

    plt.subplot(2, 2, 2)
    plt.title('Train Acc')
    plt.plot(np.arange(len(predictor_acc)), predictor_acc)
    plt.legend(['Train Acc (Best:{0:0.5f})'.format(max(predictor_acc))], loc='best')
    # plt.show()

    plt.subplot(2, 2, 3)
    plt.title('Test Loss')
    plt.ylim([0, 0.5])
    plt.plot(np.arange(len(predictor_eval_loss)), predictor_eval_loss)
    plt.legend(['Test Loss (Best:{0:0.5f})'.format(min(predictor_eval_loss))], loc='best')
    # plt.show()

    plt.subplot(2, 2, 4)
    plt.title('Test Acc')
    plt.ylim([0.5, 1.05])
    plt.plot(np.arange(len(predictor_eval_acc)), predictor_eval_acc)
    plt.legend(['Test Acc (Best:{0:0.5f})'.format(max(predictor_eval_acc))], loc='best')
    plt.savefig('result_visualization/pictures/' + name + 'loss_acc.jpg')
    plt.show()

    score_array = np.array(score_list)
    auc = ROC_PLT(score_array, label_list, 2, name, t, cv)
    aupr = PR_plt(score_array, label_list, name, t, cv)
    return auc, aupr, predictor_loss


T = 1
cv_num = [2, 5, 10]
result_dict = {}
cross_validation_loss = {}
key = 0
for t in range(T):
    print(cv_num[t], "-fold cross validation")

    auc_list = []
    aupr_list = []
    order = div_list(shuffle_indexs, cv_num[1])
    for cv in range(cv_num[1]):
        print("cross validation:", cv)
        index = order[cv]
        name = str(t) + '-' + str(cv)
        # 训练
        auc, aupr, loss = train(index, name, cv_num[t], cv)
        # 数据记录即画图
        auc_list.append(auc)
        aupr_list.append(aupr)

        if cv == 1:
            cross_validation_loss[t] = loss

    result_dict[key] = auc_list
    key += 1
    result_dict[key] = aupr_list
# filename = "result_visualization/excels/results.xlsx"
# write_excel(result_dict, filename)

# # plt.ylim(ymax=0.550, ymin=0)
# plt.plot(np.arange(len(cross_validation_loss[0])), cross_validation_loss[0],
#          label="2-fold cross valid", color='darkorange', linestyle='-', linewidth=2)
# plt.plot(np.arange(len(cross_validation_loss[1])), cross_validation_loss[1],
#          label="5-fold cross valid", color='purple', linestyle=':', linewidth=2)
# plt.plot(np.arange(len(cross_validation_loss[2])), cross_validation_loss[2],
#          label="10-fold cross valid", color='green', linestyle='-.', linewidth=2)
# plt.legend(loc="upper right")
# plt.savefig('result_visualization/pictures/x-corss_valid_loss.jpg')
# plt.show()

