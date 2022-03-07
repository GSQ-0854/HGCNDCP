import torch
from torch import nn
from torch.nn import functional as F
from utils import weight_variable_grorot1
from utils import sparse_dropout


def dot(x, y, sparse=False):
    """矩阵乘积"""
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res


class GraphConvolution(nn.Module):

    def __init__(self, size1, size2, latent_factor_num, num_features_nonzero,
                 activation=F.relu):
        super(GraphConvolution, self).__init__()

        self.size1 = size1
        self.size2 = size2
        # self.dropout = dropout
        self.activation = activation
     
        self.num_feature_nonzero = num_features_nonzero
        self.weight = nn.Parameter(torch.randn(size1[1]+size2[1], latent_factor_num))  # 权值矩阵
        self.bias = nn.Parameter(torch.randn(size1[0]+size2[0], latent_factor_num))  # 偏置矩阵
      

    def forward(self, inputs):
        """卷积操作"""
        adj, feature = inputs
  
        con = dot(adj, feature)  # (1145,1145)
        hidden = dot(con, self.weight)
        hidden = torch.add(hidden, self.bias)  # 编码后的F
        return self.activation(hidden)


class Decoder(nn.Module):
    def __init__(self, size1, size2, latent_factor_num,
                 is_spare_inputs=False):
        super(Decoder, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.is_spare_inputs = is_spare_inputs
        # self.activation = F.relu()
        self.weight = nn.Parameter(torch.randn(latent_factor_num, latent_factor_num))
        # self.weight = nn.Parameter(torch.Tensor(latent_factor_num, latent_factor_num))

    def forward(self, hidden):
        cell_num = self.size1[0]
        # drug_num = self.size2[0]
        cell_to_latent = hidden[0:cell_num, :]
        drug_to_latent = hidden[cell_num:, :]
        new_output = dot(dot(cell_to_latent, self.weight), drug_to_latent.T)
     
        return cell_to_latent, drug_to_latent, new_output
