
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from lib.gcn import GCN

class Trainer(object):
    def __init__(self, parameter, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    #更新学习率
    def update_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    #载入模型
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()

        #加载模型参数
        self.model.load_state_dict(checkpoint['model'])
        self.parameter = checkpoint['config']

    
    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.parameter,
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: 保存失败...]")
            
def unpack_batch(batch):
    inputs = [Variable(b) for b in batch[:9]]
    labels = Variable(batch[-3])
    return inputs,labels

class GCNTrainer(Trainer):
    def __init__(self, parameter, emb_matrix=None):
        self.parameter = parameter
        self.emb_matrix = emb_matrix
        self.model = GCN(parameter, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.parameters, lr=0.5, weight_decay=0)

    def update(self, batch):
        inputs, labels = unpack_batch(batch)
        self.model.train()          #开启dropout
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(inputs)
        loss = self.criterion(logits, labels)
        #正则化项
        loss_val = loss.item()
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        return loss_val

    def predict(self, batch):
        inputs,labels= unpack_batch(batch)
        self.model.eval()    #关闭dropout
        orig_idx = batch[-1]
        logits, _ = self.model(inputs)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions