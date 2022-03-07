import numpy as np
import scipy.sparse as sp
from scipy import interp
import scipy.io as scio
import random
from itertools import cycle
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import xlsxwriter as xw

def build_adj(A):
    """构建邻接矩阵"""
    M = np.zeros(shape=A.shape)
    f = open('datas/adj_n.txt', 'w')
    for row in range(M.shape[0]):
        for col in range(M.shape[1]):
            # if A[row][col] == 1:
            #     M[row][col] = 1
            #     s = str(row) + ' '+str(col)+' ' + '1' + '\n'
            #     f.write(s)
            if A[row][col] == -1:
                M[row][col] = -1
                s = str(row) + ' ' + str(col) + ' ' + '1' + '\n'
                f.write(s)
            else:
                M[row][col] = 0
    f.close()
    return M


def load_data(train_arr, test_arr):
    """加载数据集,根据传入的训练,测试序列划分数据集"""
    labels = np.loadtxt('datas/adj_p.txt')  # 读取正样本数据集
    logits_test = sp.csc_matrix((labels[test_arr, 2], (labels[test_arr, 0], labels[test_arr, 1])),
                                shape=(962, 183)).toarray()
    logits_test = logits_test.reshape([-1, 1])  # 将测试集样本转为一列

    logits_trian = sp.csc_matrix((labels[train_arr, 2], (labels[train_arr, 0],labels[train_arr, 1])),
                                 shape=(962, 183)).toarray()
    logits_trian = logits_trian.reshape([-1, 1])

    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])
    train_mask  = np.array(logits_trian[:, 0], dtype=np.bool).reshape([-1, 1])

    # 邻接矩阵拼接
    M = sp.csc_matrix((labels[train_arr, 2], (labels[train_arr, 0], labels[train_arr, 1])),
                      shape=(962, 183)).toarray()
    adj = np.vstack((np.hstack((np.zeros(shape=(962, 962), dtype=int), M)),
                     np.hstack((M.transpose(), np.zeros(shape=(183, 183), dtype=int)))))
    # adj = np.vstack((np.hstack((M, np.zeros(shape=(962, 962), dtype=int))),
    #                  np.hstack((np.zeros(shape=(183, 183), dtype=int), M.transpose()))))

    cellsim_feature = scio.loadmat('datas/cellsim.mat')['cellsim']  # (962,962)
    drugsim_feature = scio.loadmat('datas/drugsim2.mat')['drugsim2']  # (183,183)
    features = np.vstack((np.hstack((cellsim_feature, np.zeros(shape=(962, 183), dtype=int))),
                         np.hstack((np.zeros(shape=(183, 962), dtype=int), drugsim_feature))))

    features = normalize_features(features)  # 特征矩阵正则化
    adj = preprocess_adj(adj)  # 邻接矩阵正则化
    cellsim_feature_shape = cellsim_feature.shape
    drugsim_feature_shape = drugsim_feature.shape

    return adj, features, cellsim_feature_shape, drugsim_feature_shape, \
           logits_trian, train_mask, logits_test, test_mask, labels


def build_datas():
    """构建数据样本"""
    labels = scio.loadmat('datas/CL_drug_triple.mat')['CL_drug_triple']  # 邻接矩阵
    # 注：正负样本的标签值
    labels_mask = np.ones_like(labels)  # 定义标签掩码

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] == 0:
                labels_mask[i, j] = 0
                # labels[i, j] = 0   # 将标签为0的位置对应的掩码置为0 后续将转为False

    # for i in range(labels.shape[0]):
    #     for j in range(labels.shape[1]):
    #         if labels[i, j] == -1:
    #             # labels_mask[i, j] = 0
    #             labels[i, j] = 0   # 去负样本 只对正样本进行归一化操作

    adj = labels.copy()
    cellsim = scio.loadmat('datas/cellsim.mat')['cellsim']  # 细胞系相似性特征
    drugsim = scio.loadmat('datas/drugsim2.mat')['drugsim2']  # 药物相似性特征
    # cellfeaure = scio.loadmat('datas/cell_gene_express.mat')['revised_data_express']
    # drugfeature = scio.loadmat('datas/druginfo.mat')['druginfo']
    # 正样本邻接矩阵
    adj = np.vstack((np.hstack((np.zeros(shape=(962, 962), dtype=int), adj)),
                     np.hstack((adj.transpose(), np.zeros(shape=(183, 183), dtype=int)))))
    # 特征矩阵
    features = np.vstack((np.hstack((cellsim, np.zeros(shape=(962, 183), dtype=int))),
                          np.hstack((np.zeros(shape=(183, 962), dtype=int), drugsim))))
    size_cellsim = cellsim.shape  # (962,962)
    size_drugsim = drugsim.shape  # (183,183)

    # features = np.vstack((np.hstack((cellfeaure, np.zeros(shape=(962, 1444), dtype=int))),
    #                       np.hstack((np.zeros(shape=(183, 16383), dtype=int), drugfeature))))
    # size_cellsim = cellfeaure.shape  # (962,962)
    # size_drugsim = drugfeature.shape  # (183,183)
    labels_mask = np.array(labels_mask, dtype=np.bool).reshape(-1)
    adj = preprocess_adj(adj)
    features = normalize_features(features)
    return adj, features, size_cellsim, size_drugsim, labels, labels_mask


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    邻接矩阵的对称正规化
    """
    # coo_matrix函数将adj中的非0 的数进行一个张量返回 [（0,106） 1]  表示为  第0行第106列 值为1
    adj = sp.coo_matrix(adj)  # 构建张量

    # adj1 = sp.coo_mattrix(adj)

    # rowsum 也可表示为第n行的药物和rowsum个miRNA有关联
    rowsum = np.array(adj.sum(1))   # rowsum为邻接矩阵每一行的和

    # power(x,y) 求x 的y次方 flatten将多维数组降为一维
    d_inv_sqrt = np.power(abs(rowsum), -0.5).flatten()  # 输出rowsum ** -1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 溢出的部分赋值为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 对角化

    # adj = (adj*d_mat).T*d_mat   因为adj和d_mat均为对称矩阵  因此  上式等价与   d_mat * adj * d_mat
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)        # 转置后相乘
    return adj.toarray()


def normalize_features(feat):

    # 将特征矩阵求行和
    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    # 为inf设置零以避免被零除
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm


def div_list(ls, n):
    """将列表分批次"""
    ls_len = len(ls)
    j = ls_len // n  # 整数除法，返回一个不大于结果的最大整数
    ls_return = []
    for i in range(0, (n-1)*j, j):
        ls_return.append(ls[i:i+j])
    ls_return.append(ls[(n-1)*j:])
    return ls_return


def weight_variable_grorot1(intput_dim, output_dim):
    init_range = np.sqrt(6.0 / (intput_dim + output_dim))
    inital = torch.Tensor(intput_dim, output_dim).uniform_(-init_range, init_range)
    return inital

def generate_mask():
    num = 0
    adj_n = np.loadtxt('datas/adj_n.txt')
    A_N = sp.csc_matrix((adj_n[:, 2], (adj_n[:, 0], adj_n[:, 1])), shape=(962, 183)).toarray()
    A_N  = A_N.reshape([-1, 1])
    # mask = np.zeros(A_N.shape)
    # while(num < N):
    #     a = random.randint(0, 961)
    #     b = random.randint(0, 182)
    #     if A_N[a, b] == -1 and mask[a, b] != 1:
    #         mask[a,b] = 1
    #         num += 1
    mask = np.array(A_N[:, 0], dtype=np.bool).reshape([-1, 1])  # 在后面调用中会将负样本全部加入

    return mask


def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)  # noise_shape :49216
    dropout_mask = torch.floor(random_tensor).byte()

    # 获取稀疏矩阵的索引 第一组数据为行索引 第二组数据为列索引
    i = x._indices()  # [2, 49216]
    v = x._values()  # [49216]  稀疏矩阵中的值

    # [2, 49216] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]  # 取出每行中 mask非0  的索引
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1. / (1-rate))

    return out


def masked_loss(out, label, mask):
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    out = out.reshape(-1)[mask]
    label = label.reshape(-1)[mask]
    loss = criterion(out, label)
    return loss


def datas_normalize(data):
    """把值规制在0.01~max之间"""
    # mean = datas.mean()  # 均值
    # std = datas.std()    # 标准差
    # datas = (datas - mean) / std

    # d_min = 0.1
    # d_max = torch.max(datas[0]).item()
    # for data in datas:
    #     if d_max < torch.max(data).item():
    #         d_max = torch.max(data).item()
    #     else:
    #         d_max = d_max
    #
    # datas = (np.abs(datas - d_min) / (d_max - d_min))
    data = data / data.sum()
    return data


def ROC_PLT(pred, label, num_class, name, t, cv):
    label_tensor = torch.tensor(label)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    print('pred shape', pred.shape)
    print('label_onehot', label_onehot.shape)

    # 调用sklearn库 计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], pred[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # ravel()方法将数组拉成一维数组
    fpr_dict['micro'], tpr_dict['micro'], _ = roc_curve(label_onehot.ravel(), pred.ravel())
    roc_auc_dict['micro'] = auc(fpr_dict['micro'], tpr_dict['micro'])

    np.save('roc_data/all_fpr_' + str(t) + '_fold_' + str(cv) + '_cv', fpr_dict['micro'])
    np.save('roc_data/mean_tpr_' + str(t) + '_fold_'  + str(cv) + '_cv', tpr_dict['micro'])

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= num_class
        fpr_dict['macro'] = all_fpr
        tpr_dict['macro'] = mean_tpr
        roc_auc_dict['macro'] = auc(fpr_dict['macro'], tpr_dict['macro'])


        plt.figure()
        lw = 2
        plt.plot(fpr_dict['micro'], tpr_dict['micro'], label='average ROC curve (area={0:0.5f})'
                 .format(roc_auc_dict['micro']), color='deeppink', linestyle='-', linewidth=1)

        # plt.plot(fpr_dict['macro'], tpr_dict['macro'], label='macro-average ROC curve (area={0:0.2f})'
        #          .format(roc_auc_dict['macro']), color='red', linestyle='-', linewidth=1)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        # for i, color in zip(range(num_class), colors):
        #     plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
        #              label='ROC curve of class {0} (area = {1:0.5f})'
        #                    ''.format(i, roc_auc_dict[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('result_visualization/pictures/' + name + 'roc.jpg')
        plt.show()
        return roc_auc_dict['micro']


def Scatter_PLT(x, y, label):
    plt.xlabel('Resistance')
    plt.ylabel('Sensitive')
    # plt.xlim(xmax=1, xmin=0)
    # plt.ylim(ymax=1, ymin=0)
    colors1 = '#00CED1'  # 散点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4 ** 2
    x_type0 = x[np.argwhere(label == 0)[:, 0]]
    y_type0 = y[np.argwhere(label == 0)[:, 0]]
    x_type1 = x[np.argwhere(label == 1)[:, 0]]
    y_type1 = 1 - y[np.argwhere(label == 1)[:, 0]]
    plt.scatter(x_type0, y_type0, s=area, c=colors1, alpha=0.4, label='Type R')
    plt.scatter(x_type1, y_type1, s=area, c=colors2, alpha=0.4, label='Type S')
    plt.plot([0, 1], [0, 1], linewidth='0.5', color='#000000')
    plt.legend()
    plt.savefig('scatter.png', dpi=300)
    plt.show()


def Scatter_plt_3d(x, y, label):
    x0 = x[np.argwhere(label == 0)[:, 0]]
    y0 = y[np.argwhere(label == 0)[:, 0]]
    z0 = np.random.rand(len(x0), 1)
    x1 = x[np.argwhere(label == 1)[:, 0]]
    y1 = y[np.argwhere(label == 1)[:, 0]]
    z1 = np.random.randn(len(x1), 1)

    ax = plt.subplot(projection='3d')
    ax.set_title('3d_scatter_show')
    ax.scatter(x0, y0, z0, c='r', label='Type R')
    ax.scatter(x1, y1, z1, c='b', label='Type S')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig('scatter_3d.png')
    plt.show()


def softmax(x):  #
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def PR_plt(score_array, label_list, name, t, cv):
    # score_array 中存储的为[score1, score2]
    # y_pred = [score_array[i][label_list[i]] for i in range(len(label_list))]
    y_pred = [score_array[i].argmax() for i in range(len(score_array))]
    y_true = np.array(label_list)
    precision, recall, threshoulds = precision_recall_curve(y_true, y_pred)
    # precision_average = average_precision_score(y_true, y_pred, average='micro')
    aupr = auc(recall, precision)

    np.save('roc_data/recall_' + str(t) + '_fold_' + str(cv) + '_cv', recall)
    np.save('roc_data/precision_' + str(t) + '_fold_' + str(cv) + '_cv', precision)

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.plot(recall, precision, linewidth='0.5', color='b', label='AUPR={0:0.5f}'.format(aupr))
    plt.legend()
    plt.savefig('result_visualization/pictures/' + name + 'ap.jpg')

    plt.show()
    return aupr


def write_excel(data, filename):
    workbook = xw.Workbook(filename)
    worksheet = workbook.add_worksheet('sheet1')
    worksheet.activate()
    title = ['cross 0', 'cross 1', 'cross 2', 'cross 3', 'cross 4',
             'cross 5', 'cross 6', 'cross 7', 'cross 8', 'cross 9']
    col = ['2-fold auc', '2-fold aupr', '5-fold auc', '5-fold aupr',
           '10-fold auc', '10-fold aupr']
    worksheet.write_column('A2', col)
    worksheet.write_row('B1', title)
    i = 2   # 从第二行开始填写入数据
    for j in range(len(data)):
        inserData = data[j]
        row = 'B' + str(i)
        worksheet.write_row(row, inserData)
        i += 1
    workbook.close()
