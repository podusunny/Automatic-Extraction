import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import cumsum, scatter

edge_index = torch.tensor([[0, 0, 1, 2, 3],[0, 1, 0, 3, 0]])
batch1 = torch.tensor([0,0,1,1])
batch2 = torch.tensor([0,0,0,0])
#batch3 = torch.tensor([1, 1, 1, 1])
#batch4 = torch.tensor([0, 0, 0, 1])
#batch5 = torch.tensor([0, 0, 0, 0])
adj1 = to_dense_adj(edge_index,batch1)#batch1
adj2 = to_dense_adj(edge_index,batch2)#batch2
#adj3 = to_dense_adj(edge_index,batch3)#batch3
#adj4 = to_dense_adj(edge_index,batch4)#batch4
#adj5 = to_dense_adj(edge_index,batch5)#batch4
x = torch.tensor([1, 4, 1])
print("cumsum value:",cumsum(x))

print(edge_index)
#print(batch1)
print("adj1=",adj1)
#print(batch2)
print("adj2=",adj2)
#print(batch3)
#print("adj3=",adj3)
#print(batch4)
#print("adj4=",adj4)
#print(batch5)
#print("adj5=",adj5)