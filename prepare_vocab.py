#预处理训练语料

import gensim
import argparse
import json
import pickle
import copy
import numpy as np
from collections import Counter
from model import vocab, constant


def main():
    
    #输入文件
    train_file ='dataset/sem/sem.json'

    #输出文件
    vocab_file ='dataset/vocab/vocab_.pkl'
    emb_file = 'dataset/vocab/embedding_.npy'

    #加载文件
    print("正在加载语料。。。。。。")
    train_tokens = load_tokens(train_file)

    #加载word2vec
    print("正在加载GoogleNews-vectors-negative300。。。。。。")
    word2vec=gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

    #转换词向量
    print("正在转化词向量。。。。。。")
    tem_train_tokens=copy.deepcopy(train_tokens)
    for i in tem_train_tokens:
        try:
            tem_vocab = word2vec[i]
        except:
            train_tokens.remove(i)
            print("remove",i)
            
    embedding=[]
    for i in train_tokens:
        embedding.append(word2vec[i])

    embedding=np.random.uniform(0,0,(1,300)).tolist()+embedding
    v=['<UNK>']+train_tokens
    #加载glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens, glove_vocab)
    
    print("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
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
            tokens += list(filter(lambda t: t!='<PAD>', ts))
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, glove_vocab):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    v = constant.VOCAB_PREFIX + v
    print(v)
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

if __name__ == '__main__':
    main()