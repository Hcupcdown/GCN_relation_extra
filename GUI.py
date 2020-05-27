import tkinter
import threading
from lib.test import test_score 
from lib.train import train
from tkinter.filedialog import askopenfilename

class GUI:
    def __init__(self):
        
        #主窗口
        self.root_win=tkinter.Tk()
        self.root_win.geometry('600x450')
        self.root_win.resizable(width=False, height=False)
        #菜单界面
        self.mune_frame=tkinter.Frame(self.root_win)
        #训练界面
        self.whole_train_frame=tkinter.Frame(self.root_win)
        #测试界面
        self.whole_test_frame=tkinter.Frame(self.root_win)
        #左侧参数界面
        self.pare_frame=tkinter.Frame(self.whole_train_frame,width=200,height=200,padx=20)
        #右侧训练界面
        self.train_frame=tkinter.Frame(self.whole_train_frame,width=200,height=200,padx=100)
        #左侧参数界面
        self.test_pare=tkinter.Frame(self.whole_test_frame,width=200,height=200,padx=20)
        #右侧结果界面
        self.test_frame=tkinter.Frame(self.whole_test_frame,width=200,height=200,padx=100)

        #训练按钮
        self.train_button=tkinter.Button(self.mune_frame,text="训练模型",padx=15,command=self.turn_to_train)
        #测试按钮
        self.test_button=tkinter.Button(self.mune_frame,text="测试模型",padx=15,command=self.turn_to_test)
        #训练界面
        self.train_title=tkinter.Label(self.pare_frame,text="训练参数：",pady=8)
        self.gcn_num_text=tkinter.Label(self.pare_frame,text="GCN层数",pady=8)
        self.gcn_num_entry=tkinter.Entry(self.pare_frame)
        self.gcn_num_entry.insert(0, "1")
        
        self.rnn_lnum_text=tkinter.Label(self.pare_frame,text="LSTM层数",pady=8)
        self.rnn_lnum_entry=tkinter.Entry(self.pare_frame)
        self.rnn_lnum_entry.insert(0,"1")

        self.rnn_num_text=tkinter.Label(self.pare_frame,text="LSTM节点数",pady=8)
        self.rnn_num_entry=tkinter.Entry(self.pare_frame)
        self.rnn_num_entry.insert(0,"100")
 
        self.mlp_lnum_text=tkinter.Label(self.pare_frame,text="MLP层数",pady=8)
        self.mlp_lnum_entry=tkinter.Entry(self.pare_frame)
        self.mlp_lnum_entry.insert(0,"2")

        self.mlp_num_text=tkinter.Label(self.pare_frame,text="MLP节点数",pady=8)
        self.mlp_num_entry=tkinter.Entry(self.pare_frame)
        self.mlp_num_entry.insert(0,"200")

        self.lr_text=tkinter.Label(self.pare_frame,text="初始学习率",pady=8)
        self.lr_entry=tkinter.Entry(self.pare_frame)
        self.lr_entry.insert(0,"0.5")
        self.lr_decay_text=tkinter.Label(self.pare_frame,text="学习率衰减率",pady=8)
        self.lr_decay_entry=tkinter.Entry(self.pare_frame)
        self.lr_decay_entry.insert(0,"0.95")

        self.epoch_text=tkinter.Label(self.pare_frame,text="训练总趟数",pady=8)
        self.epoch_entry=tkinter.Entry(self.pare_frame)
        self.epoch_entry.insert(0,"120")

        self.bitch_text=tkinter.Label(self.pare_frame,text="单次批处理样本数",pady=8)
        self.bitch_entry=tkinter.Entry(self.pare_frame)
        self.bitch_entry.insert(0,"50")

        self.dropout_text=tkinter.Label(self.pare_frame,text="dropout",pady=8)
        self.dropout_entry=tkinter.Entry(self.pare_frame)
        self.dropout_entry.insert(0,"0.5")

        self.train_title.grid(row=0, column=0,sticky=tkinter.W)
        self.gcn_num_text.grid(row=1, column=0,sticky=tkinter.E)
        self.gcn_num_entry.grid(row=1, column=1)
        self.rnn_lnum_text.grid(row=2, column=0,sticky=tkinter.E)
        self.rnn_lnum_entry.grid(row=2,column=1)
        self.rnn_num_text.grid(row=3,column=0,sticky=tkinter.E)
        self.rnn_num_entry.grid(row=3,column=1)
        self.mlp_lnum_text.grid(row=4,column=0,sticky=tkinter.E)
        self.mlp_lnum_entry.grid(row=4,column=1)
        self.mlp_num_text.grid(row=5,column=0,sticky=tkinter.E)
        self.mlp_num_entry.grid(row=5,column=1)
        self.lr_text.grid(row=6,column=0,sticky=tkinter.E)
        self.lr_entry.grid(row=6,column=1)
        self.lr_decay_text.grid(row=7,column=0,sticky=tkinter.E)
        self.lr_decay_entry.grid(row=7,column=1)
        self.epoch_text.grid(row=8,column=0,sticky=tkinter.E)
        self.epoch_entry.grid(row=8,column=1)
        self.bitch_text.grid(row=9,column=0,sticky=tkinter.E)
        self.bitch_entry.grid(row=9,column=1)
        self.dropout_text.grid(row=10,column=0,sticky=tkinter.E)
        self.dropout_entry.grid(row=10,column=1)
       
        self.now_epoch_text=tkinter.Label(self.train_frame,text="当前趟数：0",pady=8)
        self.time_text=tkinter.Label(self.train_frame,text="耗时：0",pady=8)
        self.train_loss_text=tkinter.Label(self.train_frame,text="当前损失：0",pady=8)
        self.score_text=tkinter.Label(self.train_frame,text="训练集上F1值：0",pady=8)
        self.forget_train=tkinter.Label(self.train_frame,text="正在撤销训练......",pady=8)

        self.start_button=tkinter.Button(self.train_frame,text="开始训练",padx=15,command=self.start_train)
        self.start_button.grid(row=4,column=0,sticky=tkinter.E)
        self.training=False
        self.path=""
        #测试界面
        self.ttrain_title=tkinter.Label(self.test_pare,text="模型参数：",pady=12)
        self.tgcn_num_text=tkinter.Label(self.test_pare,text="GCN层数",pady=12)
        self.tgcn_num_entry=tkinter.Entry(self.test_pare)
        self.tgcn_num_entry.insert(0, "1")
        
        self.trnn_lnum_text=tkinter.Label(self.test_pare,text="LSTM层数",pady=12)
        self.trnn_lnum_entry=tkinter.Entry(self.test_pare)
        self.trnn_lnum_entry.insert(0,"1")

        self.trnn_num_text=tkinter.Label(self.test_pare,text="LSTM节点数",pady=12)
        self.trnn_num_entry=tkinter.Entry(self.test_pare)
        self.trnn_num_entry.insert(0,"100")
 
        self.tmlp_lnum_text=tkinter.Label(self.test_pare,text="MLP层数",pady=12)
        self.tmlp_lnum_entry=tkinter.Entry(self.test_pare)
        self.tmlp_lnum_entry.insert(0,"2")

        self.tmlp_num_text=tkinter.Label(self.test_pare,text="MLP节点数",pady=12)
        self.tmlp_num_entry=tkinter.Entry(self.test_pare)
        self.tmlp_num_entry.insert(0,"200")
        self.chose_path=tkinter.Label(self.test_pare,text = "选择模型:")

        self.ttrain_title.grid(row=0, column=0,sticky=tkinter.W)
        self.tgcn_num_text.grid(row=1, column=0,sticky=tkinter.E)
        self.tgcn_num_entry.grid(row=1, column=1)
        self.trnn_lnum_text.grid(row=2, column=0,sticky=tkinter.E)
        self.trnn_lnum_entry.grid(row=2,column=1)
        self.trnn_num_text.grid(row=3,column=0,sticky=tkinter.E)
        self.trnn_num_entry.grid(row=3,column=1)
        self.tmlp_lnum_text.grid(row=4,column=0,sticky=tkinter.E)
        self.tmlp_lnum_entry.grid(row=4,column=1)
        self.tmlp_num_text.grid(row=5,column=0,sticky=tkinter.E)
        self.tmlp_num_entry.grid(row=5,column=1)

        self.chose_button=tkinter.Button(self.test_pare, text = "选择模型", command = self.selectPath)
        self.chose_button.grid(row = 6, column = 0)
        self.path_entry=tkinter.Entry(self.test_pare, textvariable = self.path)
        self.path_entry.grid(row = 6, column = 1)
        self.stest_button=tkinter.Button(self.test_frame,text="开始测试",padx=15,command=self.start_test)
        self.stest_button.grid(row=4,column=0,sticky=tkinter.E)
        self.resoult_p=tkinter.Label(self.test_frame,pady=12)
        self.resoult_r=tkinter.Label(self.test_frame,pady=16)
        self.resoult_f1=tkinter.Label(self.test_frame,pady=16)

    #测试线程
    def stest(self,test,test2):
        parameter={
                'emb_dim':300,              #词向量维度
                'ner_dim':30,
                'pos_dim':30,
                'num_class':19,             #标签个数
                'hidden_dim':200,           #隐藏层参数个数
                'GCN_layers':1,             #GCN层数
                'rnn_layers':1,             #rnn层数
                'bidirec':True,             #rnn双向还是单项
                'rnn_dim':100,              #rnn层节点数
                'input_dropout':0.5,        #输入数据dropout率
                'word_dropout':0.04,        #词的dropout率
                'mlp_layers':1,             #多层感知器层数
                'lr':0.5,                   #sgd开始梯度下降率
                'lr_decay':0.95,            #梯度下降率
                'decay_epoch':1,            #梯度下降频率
                'num_epoch':100,            #总循环训练数
                'batch_size':50             #批处理个数
            }
        parameter['GCN_layers']=int(self.tgcn_num_entry.get())
        parameter['rnn_layers']=int(self.trnn_lnum_entry.get())
        parameter['rnn_dim']=int(self.trnn_num_entry.get())
        parameter['mlp_layers']=int(self.tmlp_lnum_entry.get())
        parameter['hidden_dim']=int(self.tmlp_num_entry.get())
        _,resoult=test_score(self.path,parameter)
        p=resoult[5:11]
        R=resoult[17:23]
        F1=resoult[30:36]
        self.resoult_p["text"]="准确率："+p
        self.resoult_r["text"]="召回率："+R
        self.resoult_f1["text"]="F1："+F1
        self.resoult_p.grid(row=0,column=0,sticky=tkinter.E)
        self.resoult_r.grid(row=1,column=0,sticky=tkinter.E)
        self.resoult_f1.grid(row=2,column=0,sticky=tkinter.E)
        self.stest_button["text"]="开始测试"
    #开始测试
    def start_test(self):
        threading._start_new_thread(self.stest,(1,2))
        self.stest_button["text"]="正在测试..."

    #选择文件路径
    def selectPath(self):
        path_ = askopenfilename()
        self.path=path_
        self.path_entry.insert(0,self.path)
        print(self.path)
    
    #跳转到训练界面
    def turn_to_train(self):
        self.mune_frame.pack_forget()
        self.whole_train_frame.pack()
        self.pare_frame.pack(side='left')
        self.train_frame.pack(side='left')

    #跳转到测试界面
    def turn_to_test(self):
        self.mune_frame.pack_forget()
        self.test_pare.pack(side='left')
        self.test_frame.pack(side='left')
        self.whole_test_frame.pack()

    #训练线程
    def strain(self,test,test2):
        parameter={
                'emb_dim':300,              #词向量维度
                'ner_dim':30,
                'pos_dim':30,
                'num_class':19,             #标签个数
                'hidden_dim':200,           #隐藏层参数个数
                'GCN_layers':1,             #GCN层数
                'rnn_layers':1,             #rnn层数
                'bidirec':True,             #rnn双向还是单项
                'rnn_dim':100,              #rnn层节点数
                'input_dropout':0.5,        #输入数据dropout率
                'word_dropout':0.04,        #词的dropout率
                'mlp_layers':1,             #多层感知器层数
                'lr':0.5,                   #sgd开始梯度下降率
                'lr_decay':0.95,            #梯度下降率
                'decay_epoch':1,            #梯度下降频率
                'num_epoch':100,            #总循环训练数
                'batch_size':50             #批处理个数
            }
        parameter['GCN_layers']=int(self.gcn_num_entry.get())
        parameter['rnn_layers']=int(self.rnn_lnum_entry.get())
        parameter['rnn_dim']=int(self.rnn_num_entry.get())
        parameter['mlp_layers']=int(self.mlp_lnum_entry.get())
        parameter['hidden_dim']=int(self.mlp_num_entry.get())
        parameter['lr']=float(self.lr_entry.get())
        parameter['lr_decay']=float(self.lr_decay_entry.get())
        parameter['num_epoch']=int(self.epoch_entry.get())
        parameter['batch_size']=int(self.bitch_entry.get())
        parameter['input_dropout']=float(self.dropout_entry.get())
        new_train=train(parameter)
        new_train.start_train()
        self.now_epoch_text.grid(row=0, column=0)
        self.time_text.grid(row=1, column=0)
        self.train_loss_text.grid(row=2, column=0)
        self.score_text.grid(row=3,column=0)
        self.start_button["text"]="结束训练"
        for i in range(1,parameter['num_epoch']+1):
            if not self.training:
                break
            train_loss,score,time=new_train.epoch_train(i)
            self.now_epoch_text["text"]="当前趟数："+str(i)
            print(train_loss)
            self.time_text["text"]="耗时："+str(time)[:10]
            self.train_loss_text["text"]="当前损失："+str(train_loss)[:10]
            self.score_text["text"]="训练集上F1值："+str(score)
        self.forget_train.grid_forget()
        self.start_button["text"]="开始训练"
    #开始训练
    def start_train(self):
        if not self.training:
            self.training=True
            threading._start_new_thread(self.strain,(1,2))
        else:
            self.training=False
            self.now_epoch_text.grid_forget()
            self.time_text.grid_forget()
            self.train_loss_text.grid_forget()
            self.score_text.grid_forget()
            self.forget_train.grid(row=3,column=0)
            self.start_button["text"]="请等待撤销完成"

    #绘制界面
    def draw(self):
        #开始训练按钮
        self.mune_frame.pack(expand=True)
        self.train_button.pack(pady=20)
        self.test_button.pack(pady=20)
        self.root_win.mainloop()
    
def main():
    gui=GUI()
    gui.draw()

if __name__ == "__main__":
    main()