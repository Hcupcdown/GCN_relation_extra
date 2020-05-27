#训练数据集
import numpy as np
import random
import torch
import time
from lib.test import test_score 
from lib.trainer import GCNTrainer
from lib.loader import DataLoader
from lib.vocab import Vocab

class train:
    def __init__(self,parameter):
        #参数字典
        self.parameter=parameter
        self.train_stop=False

    def epoch_train(self,epoch):
        if self.train_stop:
            return -1,-1,-1
        max_score=0
        start_time=time.time()
        train_loss = 0
        for i, batch in enumerate(self.train_batch):
            loss = self.trainer.update(batch)
            train_loss += loss
        #计算平均损失函数
        end_time=time.time()
        train_loss = train_loss / self.train_batch.num_examples * self.parameter['batch_size'] 
        log_str="\n第"+str(epoch)+"趟\n"+"耗时:"+str(end_time-start_time)+"\ntrain_loss:"+str(train_loss)+"\n"

        #保存模型
        self.trainer.save('./model/tem_checkpoint.pt')
        #更新梯度
        self.current_lr *= self.parameter['lr_decay']
        self.trainer.update_lr(self.current_lr)
        log_str+="current_lr:"+str(self.current_lr)+"\n"
        resoult,_=test_score('./model/tem_checkpoint.pt',self.parameter)
        score=resoult[-11:-6]
        #保存当前最好模型
        if float(score)>=max_score:
            max_score=float(score)
            #保存模型
            self.trainer.save('./model/best_model.pt')
        log_str+="\n"+resoult+"\n"
        log=open(self.log_name,"a")
        log.write(log_str)
        log.close()
        return train_loss,score,end_time-start_time

    #开始训练
    def start_train(self):
        #加载预训练语料
        vocab_file = 'dataset/vocab/vocab.pkl'
        self.vocab = Vocab(vocab_file, load=True)
        self.parameter['vocab_size'] = self.vocab.size
        emb_file = './dataset/vocab/embedding.npy'
        self.emb_matrix = np.load(emb_file)

        #加载训练集
        self.train_batch = DataLoader('dataset/sem/train_file.json', self.parameter, self.vocab, train=True)
        self.trainer = GCNTrainer(self.parameter, emb_matrix=self.emb_matrix)  
        self.current_lr = self.parameter['lr']
        self.log_name="log/"+str(time.strftime("%d_%I_%M"))+".log"
        log=open(self.log_name,'w+')
        log.write(str(self.parameter))
        log.close()
    
    #停止训练
    def stop_train(self):
        self.train_stop=True