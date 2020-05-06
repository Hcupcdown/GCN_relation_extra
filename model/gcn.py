#整个神经网络层模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.tree import Tree, sequ_to_tree, tree_to_adj
from model import constant

class GCN(nn.Module):
    def __init__(self,parameter, emb_matrix=None):
        super().__init__()
        self.parameter = parameter
        self.emb_matrix = emb_matrix
        self.layers = parameter['num_layers']
        self.mem_dim = parameter['hidden_dim']
        self.in_dim = parameter['emb_dim']
        #词嵌入层
        self.emb = nn.Embedding(parameter['vocab_size'], 300)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID),30)
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID),30)
        self.emb_matrix = torch.from_numpy(self.emb_matrix)
        #rnn层
        self.rnn = nn.LSTM(self.in_dim+60, 360, 1, batch_first=True, bidirectional=False)
        total_layers = 1
        state_shape = (total_layers, 50, 360)
        self.h0 = self.c0 = torch.autograd.Variable(torch.zeros(*state_shape),requires_grad=False)
        #载入词嵌入
        self.emb.weight.data.copy_(self.emb_matrix)
        self.gcn_drop = nn.Dropout(0.5)

        #图卷积层运算层
        self.W = nn.Linear(self.mem_dim, self.mem_dim)

        #前馈层
        layers = [nn.Linear(parameter['hidden_dim']*3, parameter['hidden_dim']), nn.ReLU()]
        for _ in range(self.parameter['mlp_layers']-1):
            layers += [nn.Linear(parameter['hidden_dim'],parameter['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
        
        #分类器
        self.classifier = nn.Linear(parameter['hidden_dim'], parameter['num_class'])

    def forward(self, inputs):
        words,_, pos, ner, head, subj_pos, obj_pos,_,l = inputs
        #对其到最大长度的句子
        maxlen = max(l)
        #将传入依存关系序列转化为树结构
        def inputs_to_tree(head, words, l,  subj_pos, obj_pos):
            head = head.cpu().numpy()
            subj_pos = subj_pos.cpu().numpy()
            obj_pos = obj_pos.cpu().numpy()
            #将依存关系序列转化为依存树
            trees=[]
            for i in range(len(l)):
                trees += [sequ_to_tree(head[i], words[i], l[i], subj_pos[i], obj_pos[i]) ]
            #adj为邻接矩阵，对齐到最大维度，添加自循环self_loop
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj)
            
        #生成邻接矩阵
        adj = inputs_to_tree(head.data, words.data, l, subj_pos.data, obj_pos.data)
        embs = [self.emb(words)]
        embs += [self.pos_emb(pos)]
        embs += [self.ner_emb(ner)]
        #组合词嵌入向量
        embs = torch.cat(embs, dim=2)
        seq_lens=list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(embs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (self.h0, self.c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)

        #生成d和掩码矩阵
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        #执行图卷积运算
        At = adj.bmm(rnn_outputs)
        AxW = self.W(At)
        AxW = AxW + self.W(rnn_outputs))
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        gcn_outputs = self.gcn_drop(gAxW)

        #池化
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        h_out = pool(gcn_outputs, mask)
        subj_out = pool(gcn_outputs, subj_mask)
        obj_out = pool(gcn_outputs, obj_mask)

        #将池化后向量拼接，生成最终的特征表达
        final_fea = torch.cat([h_out, subj_out, obj_out], dim=1)

        #通过多感知分类器输出
        outputs = self.out_mlp(final_fea)
        logits = self.classifier(outputs)
        return logits, h_out

def pool(h, mask):
    h = h.masked_fill(mask, -1e12)
    #按行取最大值
    return torch.max(h, 1)[0]

