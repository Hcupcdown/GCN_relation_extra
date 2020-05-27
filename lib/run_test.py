from lib import test
modelname="./model/best_model.pt"
#参数字典
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

print(test.test_score(modelname,parameter))