#训练数据集
import numpy as np
import random
import torch
import test

from model.trainer import GCNTrainer
from model.loader import DataLoader
from model.vocab import Vocab

#参数字典
parameter={
    'emb_dim':300,           #词向量维度
    'ner_dim':30,
    'pos_dim':30,
    'num_class':19,          #标签个数
    'hidden_dim':360,        #隐藏层参数个数
    'num_layers':1,          #全连接网络层数
    'input_dropout':0.5,     #输入数据dropout率
    'gcn_dropout':0.5,       #GCN网络dropout率
    'word_dropout':0.04,     #词的dropout率
    'mlp_layers':2,          #多层感知器层数
    'lr':0.5,                #sgd开始梯度下降率
    'lr_decay':0.95,          #梯度下降率
    'decay_epoch':1,         #梯度下降频率
    'num_epoch':150,         #总循环训练数
    'batch_size':50          #批处理个数
}

# torch.manual_seed(234)
# np.random.seed(134)
# random.seed()

#加载预训练语料
vocab_file = 'dataset/vocab/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
parameter['vocab_size'] = vocab.size
emb_file = './dataset/vocab/embedding.npy'
emb_matrix = np.load(emb_file)

#加载训练集
train_batch = DataLoader('dataset/sem/train_file.json', parameter['batch_size'], parameter, vocab, train=True)
trainer = GCNTrainer(parameter, emb_matrix=emb_matrix)  
current_lr = parameter['lr']

#开始训练
for epoch in range(1, parameter['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        loss = trainer.update(batch)
        train_loss += loss
    #计算平均损失函数
    train_loss = train_loss / train_batch.num_examples * parameter['batch_size'] 
    print("第"+str(epoch)+"趟")
    print("train_loss:"+str(train_loss))
    #保存模型
    model_file = './save_model/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    #更新梯度
    if epoch%parameter['decay_epoch']==0:
        current_lr *= parameter['lr_decay']
        trainer.update_lr(current_lr)
    print("current_lr:"+str(current_lr))
    test.test_score(epoch)
print("训练结束")