import numpy as np
#import pyshark
#import parse
import pandas as pd
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import BertModel, BertTokenizer

import model
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import Gmodel
from Graph.MinCutPool.script import utils
from sympy.physics.units import length
from torch.utils.data import DataLoader, Subset
from pandas.io.sas.sas_constants import dataset_length

from dask.sizeof import sizeof
from jsonschema.benchmarks.unused_registry import instance
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random

#from memory_profiler import profile
from scapy.all import rdpcap

from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, f1_score, roc_auc_score, log_loss

import copy
torch.autograd.set_detect_anomaly(True)

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Available GPUs:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())
print("GPU Name:", torch.cuda.get_device_name(0))
print("GPU ID:",torch.cuda.current_device())  # 当前使用的GPU编号

def save_to_txt(data, filename, mode='w'):
    """
    将数据保存为文本文件
    :param data: 要保存的数据（字符串或列表）
    :param filename: 文件名
    :param mode: 文件打开模式（'w' 或 'a'）
    """
    if not data:  # 如果数据为空，直接返回
        print("数据为空，未保存文件。")
        return

    with open(filename, mode, encoding='utf-8') as file:
        if isinstance(data[0], list):  # 如果是嵌套列表，逐行写入
            for row in data:
                file.write('\t'.join(map(str, row)) + '\n')
        elif isinstance(data, list):  # 如果是普通列表，逐行写入
            for item in data:
                file.write(str(item) + '\n')
        else:  # 如果是字符串，直接写入
            file.write(data)
def data_process(p):
    st = ''
    if isinstance(p,str):
        for t in range(0,len(p)):
            st = st + p[t]
        return st
    else:
        return 0
def addr_process(p):
    st = ''
    if isinstance(p,str):
        for t in range(0,len(p)):
            if p[t] != '.':
                st = st + p[t]
        return int(st)
    else:
        return 0
def node_node(node1,node_set):
    for n in range(0,len(node_set)):
        if node_set[n][0] == node1:
            return 1
    return 0
def genNE(src, dest, type, alln, alle, ns, sn,data_node,fulle,node_label):
    #print(str(src))
    #print(ns.items())
    if str(src) in ns:
        #print("+++++++++++++++++++++++++++++++++")
        n1 = [ns[str(src)],src]#n1 is not initial node
        if str(src) != sn[1]:#and str(dest) == sn[1]
            n1 = [sn[0],sn[1]]

    else:#n1 is initial node
        n1 = [type,src]

        ns.update({str(src):type})
    #print("n1:",n1)
    n1_1 = [n1, node_label]
    n2 = [type,dest]
    e = [n1,type,n2]

    fulle.append([n1_1,type,n2])
    if n1 not in alln and n2 not in alln:#none head and none tail
        alln.append(n1)
        alln.append(n2)
        if node_node(n1,data_node) == 1 and node_node(n2,data_node) == 1:
            alle.append(e)
        ns[str(dest)] = type
        sn[0] = type
        sn[1] = str(dest)
    elif n1 in alln and n2 not in alln:#a head and none tail
        alln.append(n2)
        if node_node(n1, data_node) == 1 and node_node(n2, data_node) == 1:
            alle.append(e)
        ns[str(dest)] = type
        sn[0] = type
        sn[1] = str(dest)
    elif n1 not in alln and n2 in alln:#none head and a tail
        alln.append(n1)
        if node_node(n1, data_node) == 1 and node_node(n2, data_node) == 1:
            alle.append(e)
        ns[str(dest)] = type
        sn[0] = type
        sn[1] = str(dest)
    elif n1 in alln and n2 in alln:#a head and a tail
        if e not in alle:
            if node_node(n1, data_node) == 1 and node_node(n2, data_node) == 1:
                alle.append(e)
            ns[str(dest)] = type
            sn[0] = type
            sn[1] = str(dest)
        else:
            ns[str(dest)] = type
            sn[0] = type
            sn[1] = str(dest)
            #print("The transition has been exist")
def edge_process(e):
    for i in range(0,len(e)):
        e[i][0][1] = int(addr_process(e[i][0][1]))
        e[i][2][1] = int(addr_process(e[i][2][1]))
        e[i] = [e[i][0][0],e[i][0][1],e[i][1],e[i][2][0],e[i][2][1]]
def Gen_edge(n_feature,adj):# generate edge feature and edge index in adjacent matrix
    X = n_feature
    src = []
    tgt = []
    for r in range(0,adj.shape[0]):
        for c in range(0,adj.shape[1]):
            if adj[r][c] == 1:
                src.append(r)
                tgt.append(c)
    edge_features = torch.cat([X[tgt],X[tgt] - X[src]], dim=-1)#X[tgt] - X[src]
    return edge_features
def Gen_edge2(n_feature,trans,e_inx):# generate edge feature and edge index in adjacent matrix
    nc = 0
    e_feature = torch.zeros(torch.sum(e_inx == 1).item(),6*n_feature.size(1))
    for r in range(0,len(e_inx)):
        if e_inx[r] == 1:
            e1 = torch.cat([n_feature[trans[1][nc]],n_feature[trans[0][nc]]],dim = 0)
            e2 = torch.zeros(2*n_feature.size(1))
            for ee in range(0,len(trans[0])):
                if trans[0][ee] == trans[1][nc] and trans[1][ee] == trans[0][nc]:
                    e2 = torch.cat([n_feature[trans[0][nc]],n_feature[trans[1][nc]]],dim = 0)
                    break
            e1 = e1.cuda()
            e2 = e2.cuda()
            e3 = e1 + e2
            e3 = e3.cuda()
            e_feature[nc] = torch.cat([e1,e2,e3],dim = 0).cuda()
            nc += 1
    return e_feature
def Dataset_Cgraph(n_feature,e_feature,e_inx):#merge node feature and edge feature, it's final train data
    count_e = 0
    count_n = 0
    feature = torch.tensor([]).cuda()#to(device)
    #print("len(e_inx) = ",len(e_inx))
    for fn in range(0,len(e_inx)):
        if e_inx[fn] == 1:
            feature = torch.cat((feature, e_feature[count_e]))
            count_e += 1
        else:
            feature = torch.cat((feature, n_feature[count_n]))
            count_n += 1
    feature = feature.view(n_feature.shape[0]+e_feature.shape[0],e_feature.shape[1])
    return feature
def Dataset_Cgraph2(n_feature,edge_num,e_inx):#merge node feature and placeholder of edge feature
    count_e = 0
    count_n = 0
    feature = torch.tensor([])
    e_feature = torch.zeros(n_feature.shape[1])
    #print("len(e_inx) = ",len(e_inx))
    for fn in range(0,len(e_inx)):
        if e_inx[fn] == 1:
            feature = torch.cat((feature, e_feature))
            count_e += 1
        else:
            feature = torch.cat((feature, n_feature[count_n]))
            count_n += 1
    feature = feature.view(n_feature.shape[0]+edge_num,n_feature.shape[1])
    return feature
def search(path,filen,fileh):
    content = os.listdir(path)
    for each in content:
        each_path = path + os.sep + each
        filen.append(each_path)
        fileh.append(each)
        #print(each_path)
        '''
		if os.path.isdir(each_path):
			search(each_path)
        '''
def EdgeMatch(e1,e2,index):
    buff = []
    for i in range(0,len(e1)):
        for j in range(0,len(e1)):
            #print("edge=",[e1[i][0],e1[j][0][0],e1[j][0]])
            #print("acce=",e2[2])
            if [e1[i][0],e1[i][1],e1[j][0],e1[j][0],e1[j][1]] in e2:
                if [e1[i][0],e1[i][1],e1[j][0],e1[j][0],e1[j][1]] in buff:
                    a=0
                    index.append(a)
                else:
                    a=1
                    index.append(a)
                    buff.append([e1[i][0], e1[i][1], e1[j][0], e1[j][0], e1[j][1]])
            else:
                a=0
                index.append(a)

    return index
#node_l format [[type,src],label] edge_l format [node_l,type,node_l]
def Edge_to_Node(node_l,edge_l):
    adj_node = np.zeros((len(node_l),len(node_l)))
    co = 0
    abd = []
    for ed in range(0,len(edge_l)):
        la1 = 0
        la2 = 0
        l1 = 0
        l2 = 0
        lab = 0
        for dl in range(0,len(node_l)):
            if edge_l[ed][0] == node_l[dl][0] and l1 == 0:
                la1 = node_l[dl][1]
                #print("edge_node1:",edge_l[ed][0],node_l[dl][0])
                l1 = 1

            if edge_l[ed][2] == node_l[dl][0] and l2 == 0:
                la2 = node_l[dl][1]
                #print("edge_node2:", edge_l[ed][0], node_l[dl][0])
                l2 = 1

            if l1 == 1 and l2 == 1:
                adj_node[la1][la2] = 1
                lab = 1
                break
        #if lab == 0:
            #print("edge_node:", edge_l[ed])
        #if [la1,la2] in abd:
            #print("ssss:",[la1,la2])

        abd.append([la1,la2])
        co += 1
    #print("node_l = ",node_l)
    #print("edge_l = ", edge_l)
    #print("co:", co)
    #print("abd:",len(abd), abd)
    #print("np.sum(adj_node == 1):", np.sum(adj_node == 1))
    return adj_node
#edge_l format [[node_l,label],type,node_l],
def Edge_to_Node2(node_l,edge_l):
    adj_node = np.zeros((len(node_l),len(node_l)))
    buffern = []
    buffern2 = []
    c1= 0
    c2 = 0
    for ed in range(0,len(edge_l)):
        for dl in range(0,len(node_l)):
            if edge_l[ed][2] == node_l[dl][0]:# and [edge_l[ed][0],edge_l[ed][2]] not in buffern2
                if ed >0 and dl <= ed:
                    adj_node[ed, dl] = 1
                    c2 += 1
                    if edge_l[ed][0][0] in buffern:
                        for cv in range(0, ed):
                            if edge_l[cv][0][0] == edge_l[ed][0][0]:
                                adj_node[cv, dl] = 1
                                c1 += 1
                                #print("c1", c1, cv, dl)
                                break
                    #real edge
                    #elif [edge_l[ed][0][0],edge_l[ed][2]] not in buffern2:
                    #    adj_node[ed, dl] = 1
                    #    c2 += 1
                    #    print("c2",c2,ed, dl)
                    buffern2.append([edge_l[ed][0][0],node_l[dl][0]])
                break

        buffern.append(edge_l[ed][0][0])
    #print(f"c1={c1} c2={c2}")
    return adj_node
def adj_to_adj(adj_matrix,coord,label=0,weight0=0):
    ones_count = 0#np.sum(adj_matrix == 1)
    extend_adj = np.zeros((adj_matrix.shape[0]+np.sum(adj_matrix == 1),adj_matrix.shape[1]+np.sum(adj_matrix == 1)))
    #print("np.sum(adj_matrix == 1):",np.sum(adj_matrix == 1))
    extend_node_index = np.zeros(adj_matrix.shape[0]+np.sum(adj_matrix == 1))
    extend_node = []#np.zeros(adj_matrix.shape[0]+np.sum(adj_matrix == 1))
    for r in range(1,adj_matrix.shape[0]+1):
        for c in range(1, adj_matrix.shape[1]+1):
            if adj_matrix[r-1][c-1] == 1:
                ones_count += 1
                if c-1 == 0:
                    tc = 0#np.sum(adj_matrix[0:c,0:adj_matrix.shape[1]] == 1)#row adj_matrix slicing
                else:
                    tc = np.sum(adj_matrix[0:c - 1, 0:adj_matrix.shape[1]] == 1)  # row adj_matrix slicing
                if r-1 == 0:
                    tr = 0#np.sum(adj_matrix[0:r, 0:adj_matrix.shape[1]] == 1)  # column adj_matrix slicing
                else:
                    tr = np.sum(adj_matrix[0:r - 1, 0:adj_matrix.shape[1]] == 1)  # column adj_matrix slicing
                if label == 1 and [r-1,c-1] in coord:
                    extend_node_index[ones_count+r-1] = 0
                    #print([r-1,c-1])
                else:
                    extend_node_index[ones_count+r-1] = 1
                if weight0 ==1:
                    extend_adj[r+tr-1][ones_count+r-1] = np.sum(adj_matrix == 1) + 1 -ones_count#extended node for edge
                    extend_adj[ones_count+r-1][c+tc-1] = np.sum(adj_matrix == 1) + 1 -ones_count
                else:
                    extend_adj[r+tr-1][ones_count+r-1] = 1#extended node for edge
                    extend_adj[ones_count+r-1][c+tc-1] = 1
                extend_node.append([r+tr-1,ones_count+r-1])
                extend_node.append([ones_count+r-1,c+tc-1])

    return extend_adj, extend_node_index, extend_node
def label_mat(inx,num):
    mat = np.zeros((num,len(inx)))
    l = 0
    for o in range(0,len(inx)):
        if inx[o] == 1:
            mat[l][o] = 1
            l += 1
    return mat
def check_adj_match(adj_std,adj_noise):
    a = 0
    if adj_std.shape != adj_noise.shape:
        return 0
    for r in range(0,adj_std.shape[0]):
        for c in range(0,adj_std.shape[1]):
            #print("adj eq=", adj_std[r][c], adj_noise[r][c])
            if adj_std[r][c] == 1 and adj_noise[r][c] == 1:
                a+=1
            elif adj_std[r][c] == 1 and adj_noise[r][c] == 0:
                print(r,c)
                return 0
            else:
                pass

    return 1
def adj_match(adj_std,adj_noise):#adj_noise obtain adj_std
    coord_xy = []
    for r in range(0,adj_std.shape[0]):
        for c in range(0,adj_std.shape[1]):
            if adj_noise[r][c] == 1 and adj_std[r][c] == 0:
                #print([r,c])
                coord_xy.append([r,c])
    return coord_xy
def adj_process(adj):
    src = []
    dest = []
    trans = []
    for r in range(0,adj.shape[0]):
        for c in range(0,adj.shape[1]):
            if adj[r][c] == 1:
                src.append(r)
                dest.append(c)
    trans.append(src)
    trans.append(dest)
    return trans


class AllModel(nn.Module):
    def __init__(self,BERT_PATH,input_dim, hidden_dim, output_dim):
        super(AllModel, self).__init__()

        self.bert_model = BertModel.from_pretrained(BERT_PATH)

        self.GCN = Gmodel.GcnModel(input_dim, hidden_dim, output_dim)

        self.norm = nn.LayerNorm(input_dim)
        self.norm_edge = nn.LayerNorm(data_length)
        # 定义 MLP，用于边特征计算
        self.mlp = nn.Sequential(

            nn.Linear(6*input_dim, hidden_dim),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, data, data_mask, e_adj_matrix,transition,label_inx_positive,label_inx,mask):
        #print("data:",data.size(),data)
        #print("data_mask:", data_mask.size(), data_mask)
        sequence_output = self.bert_model(input_ids=data,attention_mask=data_mask)

        GNN_edges = Gen_edge2(sequence_output.pooler_output, transition, label_inx_positive)

        #print("node:",sequence_output.pooler_output.size(),sequence_output.pooler_output)
        GNN_edges = GNN_edges.cuda()
        GNN_edges = self.mlp(GNN_edges)
        #print("GNN_edges:", GNN_edges.size(), GNN_edges)
        GNN_data = Dataset_Cgraph(sequence_output.pooler_output, GNN_edges, label_inx_positive)
        GNN_data = GNN_data.cuda()
        #print("GNN_data:", GNN_data.size(), GNN_data)
        #print("GNN_inputs:", GNN_inputs.size(),GNN_inputs)
        train_all, loss_a, da = self.GCN(GNN_data,e_adj_matrix,label_inx,mask)
        return train_all, loss_a

# 遍历所有分组
if os.name == 'nt':
    #tokenizer = LEDTokenizer.from_pretrained(os.getcwd()+"\\local_led_tokenizer") #AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained(os.getcwd() + "\\bert-base-uncased")
    BERT_PATH = os.getcwd() + "\\bert-base-uncased"
else:
    #tokenizer = LEDTokenizer.from_pretrained(os.getcwd() + "//local_led_tokenizer")
    tokenizer = BertTokenizer.from_pretrained(os.getcwd() + "//bert-base-uncased")
    BERT_PATH = os.getcwd() + "//bert-base-uncased"
cuda_condition = torch.cuda.is_available()
#device = torch.device("cuda:1" if cuda_condition else "cpu")
#print('device',device)
filen = []
fileh = []
filen_valid = []
fileh_valid = []
tolerance = 0.0001
accumulationcc = 0
batch_mqtt = 1000
data_length = 20
data_dim = 768#hidden_size*data_length
hidden_dim = 768#data_dim
output_dim = 2
num = 0.05
accum_steps = 20
ind = []
run_model = AllModel(BERT_PATH, data_dim, hidden_dim, output_dim)
run_model = run_model.cuda()

if torch.cuda.device_count() > 1:
    print(f"use {torch.cuda.device_count()} GPUs")
    run_model = nn.DataParallel(run_model)
else:
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    run_model.to(device)
    print('one device',device)

optimizer = optim.Adam(run_model.parameters(), lr=1e-4)#,weight_decay=1e-5
if os.name == 'nt':
    search(os.getcwd() + '\\train_set',filen,fileh)
    search(os.getcwd() + '\\valid_set', filen_valid, fileh_valid)
else:
    search(os.getcwd() + '//train_set',filen,fileh)
    search(os.getcwd() + '//valid_set', filen_valid, fileh_valid)

print(torch.__version__)
Log = open('log.txt', 'w', encoding='utf-8')
# 获取原始字节数据

#print(fileh)
random.seed(20)
cc = 0
strr = ""
epoch_batch = []
epoch = 1
loss_value1 = 0
loss_value2 = 0

#train set

for name in range(0,int(len(fileh))):#1):#
    count_buffer = 0
    pre_loss = 0
    lossloss1 = 0
    lossloss2 = 0
    count_steps = 0
    acc_mean = []
    F1_mean_macro = []
    if os.name == 'nt':
        packets = rdpcap(os.getcwd()+"\\train_set\\"+fileh[name])
        #packets = rdpcap(os.getcwd()+"\\train_set\\" + fileh[0])
        #packets = rdpcap(os.getcwd() + "\\train_set\\NormalData1.pcap")
    else:
        packets = rdpcap(os.getcwd()+"//train_set//"+fileh[name])
        #packets = rdpcap(os.getcwd()+"//train_set//" + fileh[0])
        #packets = rdpcap(os.getcwd() + "//train_set//NormalData1.pcap")
    run_model.train()
    print("len(packets)：",len(packets), (count_buffer + batch_mqtt))
    while (len(packets) > (count_buffer + batch_mqtt)):#for count in range(0, int(len(packets) / batch_mqtt)):
        label = 0
        node_label = []
        mqtte = []
        mqttn = []
        data = []
        node = []
        edge = []
        edge_full = []
        adj = np.zeros((batch_mqtt,batch_mqtt))
        ns = {}
        cn = [0, '']
        print("count_buffer=", count_buffer, count_buffer + batch_mqtt, batch_mqtt)
        for packet in packets[ count_buffer : count_buffer + batch_mqtt]:
            #parse MQTT
            if packet.haslayer('Raw'):
                cc += 1
                layer = packet['Raw'].load
                src = data_process(packet['IP'].src)
                strr = packet['IP'].src  # str(hex(src))
                dst = data_process(packet['IP'].dst)
                # print("dst=",dst)
                strr += packet['IP'].dst  # str(hex(dst))
                # print('MQTT data flow:', layer)
                raw_bytes = bytes(layer)
                for j in range(0, len(raw_bytes)):
                    strr += str(hex(raw_bytes[j]))
                #print("strr:",strr)
                tokens = tokenizer.tokenize(strr)
                ids = tokenizer.convert_tokens_to_ids(tokens)


                #print(len(tokens),tokens)
                #print(len(ids),ids)
                # get accurate graph
                fun = int(hex(raw_bytes[0] >> 4), 16)

                if str(src) in ns:# n1 is not initial node
                    #node1 is a traffic label
                    node1 = [[ns[str(src)], src], label]
                    if str(src) != cn[1]:  # and str(dest) == sn[1]
                        node1 = [[cn[0], cn[1]], label]
                else:  # n1 is initial node
                    node1 = [[fun, src], label]

                node2 = [[fun, dst], label]

                #print("ids len:", len(ids),ids)
                #unify data length


                if len(ids) > data_length:
                    ids = torch.LongTensor(ids[0:data_length])

                #e = [[node1, ids, node2], label]
                #ee = [[idnode1], [ids], [idnode2]]
                mqtte.append(node1)  # e = [[node1, ids, node2], label], ee = [[idnode1, ids, idnode2],[label]]

                #edge_full:label of fully edges
                genNE(src, dst, fun, node, edge, ns, cn,mqtte,edge_full,label)

                data.append(torch.LongTensor(ids))
                node_label.append([fun, addr_process(dst), label])
                label += 1

        print("This is train set:",fileh[name],"batch_mqtt:",batch_mqtt,"real edge num:", len(edge))

        adj_fully = Edge_to_Node2(mqtte, edge_full)
        adj = Edge_to_Node(mqtte, edge)
        adj_trans = adj_process(adj_fully)
        x_y = []
        extend_adj, extend_adj_index, extend_node_index = adj_to_adj(adj,x_y)
        fe_x_y = adj_match(adj,adj_fully)#coord.
        check_adj = check_adj_match(adj, adj_fully)
        #print("fe_x_y",len(fe_x_y),check_adj)
        if check_adj != 1:
            raise ValueError("Not Match")
        adj_e = adj_fully
        options = ["A", "B"]
        weights = [0, 1]  # A  20%，B 80%
        choice = random.choices(options, weights=weights, k=1)[0]
        addE = 0
        adj_fully_len = np.sum(adj_fully == 1)
        adj_len = np.sum(adj == 1)

        num = random.uniform(1.5,2.5)#1.8, 2.8
        if choice == "A":
            #for h in range(0,int((adj_fully_len-len(edge))*2)):# adj_fully_len-1.5*len(edge)
            while addE < int((adj_fully_len-len(edge)*num)):
                adj_x =random.randint(0, batch_mqtt-1)
                adj_y= random.randint(0, batch_mqtt-1)
                if adj_e[adj_x][adj_y] == 1 and adj[adj_x][adj_y] == 0:
                    addE += 1
                    adj_e[adj_x][adj_y] = 0
                    fe_x_y.remove([adj_x, adj_y])
            extend_adj2, extend_adj_index2, extend_node_index2 = adj_to_adj(adj_e,fe_x_y, 1,0)
            extend_adj3, extend_adj_index3, extend_node_index3 = adj_to_adj(adj_e, fe_x_y)
            mask_mat = label_mat(extend_adj_index3, np.sum(adj_fully == 1) - addE)
        else:
            extend_adj2, extend_adj_index2, extend_node_index2 = adj_to_adj(adj_e, fe_x_y, 1,0)
            extend_adj3, extend_adj_index3, extend_node_index3 = adj_to_adj(adj_e, fe_x_y)
            mask_mat = label_mat(extend_adj_index3, np.sum(adj_fully == 1))



        mask_inx = []
        mask_rev = []
        for cc in range(0,len(extend_adj_index3)):
            if extend_adj_index3[cc] == 1:
                mask_inx.append(extend_adj_index2[cc])
                if extend_adj_index2[cc] == 1:
                    mask_rev.append([0,1])
                else:
                    mask_rev.append([1,0])

        mqtt_tensor = pad_sequence(data, batch_first=True, padding_value=0)


        mqtt_tensor = mqtt_tensor.detach().cpu()
        #mqtt_tensor = Dataset_Cgraph2(mqtt_tensor, adj_fully_len, extend_adj_index3)
        data_mask = (mqtt_tensor != 0).bool()

        print(f"mqtt_tensor size:{mqtt_tensor.size()} data_mask:{data_mask.size()}")
        #print(data_mask)
        #print(mqtt_tensor)
        mqtt_tensor = mqtt_tensor.long().cuda()
        data_mask = data_mask.long().cuda()

        inx = extend_adj_index3
        inx2 = extend_adj_index2
        inx3 = mask_inx
        inx4 = mask_rev
        inx = torch.LongTensor(inx).cuda()
        inx2 = torch.LongTensor(inx2).cuda()
        inx3 = torch.LongTensor(inx3).cuda()
        inx4 = torch.LongTensor(inx4).cuda()


        mask_mat = torch.LongTensor(mask_mat).cuda()
        adj = torch.LongTensor(adj).cuda()
        adj_e = torch.LongTensor(adj_e).cuda()
        extend_adj = torch.LongTensor(extend_adj).cuda()
        extend_adj2 = torch.LongTensor(extend_adj2).cuda()

        train_result,loss_loss = run_model.forward(mqtt_tensor,data_mask,extend_adj2,adj_trans,inx,inx4,mask_mat)
        #print(f"epoch:{epoch} train_result:{train_result.shape} {train_result}")

        train_result = train_result.detach().cpu().numpy()

        predicted_labels = train_result[:,1]
        for i in range(0, len(predicted_labels)):
            if predicted_labels[i] >= 0.5:
                predicted_labels[i] = 1.0
            else:
                predicted_labels[i] = 0.0

        ma = torch.LongTensor(mask_inx)
        predicted_labels = torch.LongTensor(predicted_labels)
        predicted_labels = torch.squeeze(predicted_labels,0)

        #print(f"predicted_labels: {predicted_labels}")
        valid_acc = accuracy_score(ma,predicted_labels,normalize=True)
        correct_count = accuracy_score(ma, predicted_labels, normalize=False)
        f1_4 = f1_score(ma, predicted_labels, average='weighted')
        f1_3 = f1_score(ma, predicted_labels, average='macro')
        f1_2 = f1_score(ma, predicted_labels, average='micro')
        acc_mean.append(valid_acc)
        F1_mean_macro.append(f1_3)
        print(f"accuracy:{valid_acc} acc_mean:{sum(acc_mean)/len(acc_mean)} correct count:{correct_count}")
        print(f"F1-weighted:{f1_4}")
        print(f"F1-macro:{f1_3}")
        print(f"F1-micro:{f1_2}")
        print(f"epoch: {epoch}",f"Train set crossEntropy Loss: {loss_loss}")
        print(f"e_adj edges num:{adj_fully_len} adj edges num:{len(edge)}")
        loss_loss.backward()

        count_steps += 1
        if count_steps == accum_steps:
            optimizer.step()
            optimizer.zero_grad()
            count_steps = 0



        lossloss1 = loss_loss.item()
        lossloss2 = abs(loss_loss.item() - pre_loss)
        #if abs(loss_loss.item() - pre_loss) < tolerance:
        #    print(f"Converged at epoch {epoch} with loss {loss_loss.item()}")
        #    break

        pre_loss = loss_loss.item()
        min_val = 800
        max_val = 1200
        batch_mqtt = random.randint(min_val, max_val)
        count_buffer = count_buffer + batch_mqtt

    if count_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        count_steps = 0

    Log.write("epoch:" +str(epoch)+ " ")
    Log.write("acc_mean:" + str(sum(acc_mean)/len(acc_mean)) + " ")
    Log.write("F1_mean_macro:" + str(sum(F1_mean_macro)/len(F1_mean_macro)) + " ")
    Log.write("loss:" + str(lossloss1)+ str(lossloss2) + " ")

    epoch += 1

    pre_model = copy.deepcopy(run_model)
    torch.save(pre_model, "Pre_TrGcn_model.pth")

