from stanfordcorenlp import StanfordCoreNLP
import json
import pickle 
import numpy as np
from collections import Counter
from lib import vocab
from lib import constant

#将语法依存结构转换为序列
def trans(raw_list):
    resoult=[0 for i in range(len(raw_list))]
    for i in raw_list:
        resoult[i[2]-1]=i[1]
    return resoult

#转换为json格式，并用语法分析工具生成语法依赖树
def pretreatment(readfile_name,writefile_name):
    nlp = StanfordCoreNLP(r'./pretreatment/stanfordnlp/stanford-corenlp-full-2018-10-05',lang='en')
    sentence="people have been moving back into downtown"
    print(nlp.dependency_parse(sentence))
    file_read=open(readfile_name)
    file_write=open(writefile_name,"w")
    resoult=[]
    raw_line=file_read.readline()
    counter=0
    print("正在生成依存关系序列...")
    while raw_line:
        tem_dir={}
        ori_sent=raw_line[raw_line.find('"')+1:-3]
        tokens=nlp.word_tokenize(ori_sent)
        subj_start=tokens.index('<e1>')
        del tokens[subj_start]
        subj_end=tokens.index('</e1>')-1
        del tokens[subj_end+1]
        obj_start=tokens.index('<e2>')
        del tokens[obj_start]
        obj_end=tokens.index('</e2>')-1
        del tokens[obj_end+1]
        relation=file_read.readline()
        tem_dir['id']=counter
        relation=relation[:-1]
        tem_dir['relation']=relation
        tem_dir['tokens']=tokens
        tem_dir['subj_start']=subj_start
        tem_dir['subj_end']=subj_end
        tem_dir['obj_start']=obj_start
        tem_dir['obj_end']=obj_end
        line=" ".join(tokens)
        stanford_pos=nlp.pos_tag(line)
        stanford_ner=nlp.ner(line)
        tem_dir['stanford_pos']=[p[1] for p in stanford_pos ]
        tem_dir['stanford_ner']=[p[1] for p in stanford_ner]
        tem_dir['stanford_head']=trans(nlp.dependency_parse(line))
        counter+=1
        resoult.append(tem_dir)
        _=file_read.readline()
        _=file_read.readline()
        raw_line=file_read.readline()
    json.dump(resoult,file_write,ensure_ascii=False,indent=4)
    file_read.close()
    file_write.close()

def establish_vocab():
    #输入文件
    train_file ='dataset/sem/train_file.json'
    test_file='dataset/sem/test_file.json'
    #词向量工具所在文件夹
    wv_file = 'pretreatment/glove/glove.840B.300d.txt'
    #词向量维度：300
    wv_dim = 300

    #输出文件
    vocab_file ='dataset/vocab/vocab.pkl'
    emb_file = 'dataset/vocab/embedding.npy'

    #加载文件
    print("加载训练集与测试集...")
    tokens = load_tokens(train_file)
    tokens+=load_tokens(test_file)

    #加载glove
    print("加载Glove词向量...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    
    #构建词汇表
    v = build_vocab(tokens, glove_vocab)
    
    #构建词向量
    embedding = vocab.build_embedding(wv_file, v, wv_dim)

    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    print("all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            ts = d['tokens']
            tokens += ts
    return tokens

def build_vocab(tokens, glove_vocab):
    counter = Counter(t for t in tokens)
    v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    v = constant.VOCAB_PREFIX + v
    return v

def main():
    #处理训练集
    pretreatment("./dataset/raw_sem/TRAIN_FILE.TXT","./dataset/sem/train_file.json")
    #处理测试集
    pretreatment("./dataset/raw_sem/TEST_FILE_FULL.TXT","./dataset/sem/test_file.json")
    #生成词向量
    establish_vocab()

if __name__ == '__main__':
    main()
    