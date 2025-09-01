"""MinCutPool模型结构
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Graph.MinCutPool.script import layers
import utils

#from .utils import *
#from .layers import *
from math import ceil

from sympy import false
from torch.functional import Tensor
from torch.onnx.symbolic_opset9 import tensor
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dense_batch
#device = torch.device("cuda:0" if cuda_condition else "cpu")


class GcnModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                dropout=0.1, use_bias=True):
        """
            Inputs:
            -------
            input_dim: int, 节点特征数量
            hidden_dim: int, 各隐层计算输出的特征数
            output_dim: int, 输出类别数量
            avg_nodes: int, 所有图平均节点数, 用于确定各聚类层的聚类个数
            dropout: float, 输出层使用的dopout比例
            use_bias: boolean, 图卷积层是否使用偏置

        """

        super(GcnModel, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.act = nn.ReLU(inplace=True)
        self.act_SM = nn.Softmax(dim=1)
        self.norm = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(p=dropout)
        # 图卷积层
        self.convs1 = [layers.GraphConvolution(input_dim, 2 * input_dim, use_bias) for _ in range(20)]
        self.convs2 = [layers.GraphConvolution(2 * input_dim, input_dim, use_bias) for _ in range(20)]

        # 节点聚类层
        self.cluster = nn.Linear(hidden_dim, hidden_dim)
        #self.cluster1 = nn.Linear(hidden_dim, hidden_dim)

        # 节点输出层

        #edge feed forward
        self.ffd = nn.Sequential(

            nn.Linear(input_dim, ceil(hidden_dim)),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(ceil(hidden_dim), output_dim),

        )
        return


    def forward(self, data1,data2,inx,mask):

        data_F, graph = data1, data2
        a0 = graph.float()

        x = data_F
        for ii in range(20):
            x0 = x
            x = self.convs1[ii](a0,x)
            x = self.act(x)
            x = self.drop(x)
            x = self.convs2[ii](a0,x)
            x = self.norm(x0 + self.drop(x))
        mask = mask.float()
        datax = x
        logits = self.ffd(x)
        rs = torch.mm(mask,logits).clip(min=1e-6)

        inx = inx.float()

        loss_g = self.criterion(rs, inx)

        rs = self.act_SM(rs)

        #print("rs_p:", loss_g,len(rs_p),rs_p)
        #print("rs:",len(rs[:,1]),rs[:,1])
        #print("inx:",torch.sum(inx == 1).item(), len(inx),inx[:,1])
        return rs, loss_g, datax
