import json
import random
import torch
import numpy as np

from lib import vocab

class DataLoader(object):

    def __init__(self, filename, parameter, vocab, train=False):
        self.batch_size = parameter['batch_size']
        self.parameter = parameter
        self.vocab = vocab
        self.train = train
        self.label2id = {'Other': 0, 'Cause-Effect(e1,e2)': 1, 'Cause-Effect(e2,e1)': 2, 'Component-Whole(e1,e2)': 3,
                        'Component-Whole(e2,e1)': 4, 'Content-Container(e1,e2)': 5, 'Content-Container(e2,e1)': 6, 
                        'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8, 'Entity-Origin(e1,e2)': 9, 
                        'Entity-Origin(e2,e1)': 10, 'Instrument-Agency(e2,e1)': 11, 'Instrument-Agency(e1,e2)': 12, 
                        'Member-Collection(e1,e2)': 13, 'Member-Collection(e2,e1)': 14, 'Message-Topic(e1,e2)': 15, 
                        'Message-Topic(e2,e1)': 16, 'Product-Producer(e1,e2)': 17, 'Product-Producer(e2,e1)': 18}

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.get_all_data(data, vocab)

        #训练时将数据集打乱
        if train:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        #封装成批
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.data = data

    #处理数据
    def get_all_data(self, data, vocab):

        POS_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 
                    'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 
                    'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 
                    'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 
                    'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, 
                    '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 
                    'LS': 44, 'UH': 45, '#': 46}
        NER_TO_ID= {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 
                    'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 
                    'ORDINAL': 12, 'TIME': 13, 'SET': 14}
        all_data = []
        for d in data:
            tokens = list(d['tokens'])
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], NER_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            relation = self.label2id[d['relation']]
            all_data += [(tokens, pos, ner, head, subj_positions, obj_positions, relation)]
        return all_data

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        lens = [len(x) for x in batch[0]]

        # word dropout
        if self.train:
            words = [word_dropout(sent, self.parameter['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        #将类型转换为tensor
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        head = get_long_tensor(batch[3], batch_size)
        subj_positions = get_long_tensor(batch[4], batch_size)
        obj_positions = get_long_tensor(batch[5], batch_size)

        rels = torch.LongTensor(batch[6])
        l=torch.IntTensor(lens)
        return (words, masks, pos, ner, head, subj_positions, obj_positions, rels,l,orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else 1 for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    #获取宾语主语位置并转换为list
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    #转换为long_tensor
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    return [1 if x != 1 and np.random.random() < dropout \
            else x for x in tokens]

