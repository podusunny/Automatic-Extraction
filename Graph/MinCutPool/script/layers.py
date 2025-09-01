
import torch
import torch.nn as nn
#import torch_scatter

# ----------------------------------------------------------------------------
# 图卷积层


class GraphConvolution(nn.Module):
    """图卷积层
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积层

            Inputs:
            -------
            input_dim: int, 输入特征维度
            output_dim: int, 输出特征维度
            use_bias: boolean, 是否使用偏置

        """

        super(GraphConvolution, self).__init__()

        self.use_bias = use_bias

        self.weight = nn.Linear(input_dim, output_dim)



        return

    def forward(self, adjacency, X):
        """图卷积层前馈

            Inputs:
            -------
            adjacency: tensor in shape [batch_size, num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [batch_size, num_nodes, input_dim], 节点特征

            Output:
            -------
            output: tensor in shape [batch_size, num_nodes, output_dim], 输出

        """

        weight = self.weight#torch.cat([self.weight.unsqueeze(0)] * X.size(0))
        X = X.cuda()
        weight = weight.cuda()
        #torch.mm(X, weight)
        adj = torch.diag(torch.Tensor([1]*adjacency.size()[0]).cuda()).cuda() + adjacency
        adj = adj.cuda()
        D_inv_sqrt = torch.diag(1 / torch.sqrt(adj.sum(dim=1)).cuda()).cuda()
        D_inv_sqrt = D_inv_sqrt.cuda()
        a1 = torch.mm(adj, D_inv_sqrt).cuda()
        a1 = a1.cuda()
        a2 = torch.mm(D_inv_sqrt, a1)
        a2 = a2.cuda()
        output = torch.mm(a2, X)
        output = output.cuda()
        output = weight(output)
        output = output.cuda()
        #print("adjacency size:",adjacency.size())
        #print("weight size:", weight.size(), "X size:", X.size(),"support size:",support.size(),"output size:",output.size())
        '''
        self.use_bias = self.use_bias
        if self.use_bias:
            output = output + self.bias
            output = output.cuda()
        '''

        return output

