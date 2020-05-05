from stanfordcorenlp import StanfordCoreNLP
import json
import os				# 导入os模块

nlp = StanfordCoreNLP(r'./pretreatment/stanford-corenlp-full-2018-10-05',lang='en')

#将语法依存结构转换为序列
def trans(raw_list):
    resoult=[0 for i in range(len(raw_list))]
    for i in raw_list:
        resoult[i[2]-1]=i[1]-1
    return resoult

#转换为json格式，并用语法分析工具生成语法依赖树
def pretreatment(readfile_name,writefile_name):
    file_read=open(readfile_name)
    file_write=open(writefile_name,"w")
    resoult=[]
    raw_line=file_read.readline()
    counter=0
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
        tem_dir['subj_type']=nlp.ner(tokens[subj_start])[0][1]
        tem_dir['obj_type']=nlp.ner(tokens[obj_start])[0][1]
        line=" ".join(tokens)
        stanford_pos=nlp.pos_tag(line)
        stanford_ner=nlp.ner(line)
        tem_dir['stanford_pos']=[p[1] for p in stanford_pos ]
        tem_dir['stanford_ner']=[p[1] for p in stanford_ner]
        tem_dir['stanford_head']=trans(nlp.dependency_parse(line))
        tem_dir['stanford_deprel']=[]
        counter+=1
        resoult.append(tem_dir)
        _=file_read.readline()
        _=file_read.readline()
        raw_line=file_read.readline()
        print(counter-1)
    json.dump(resoult,file_write,ensure_ascii=False,indent=4)
    file_read.close()
    file_write.close()

def main():
    #处理训练集
    pretreatment("./pretreatment/SemEval/TEST_FILE_FULL.TXT","./dataset/sem/test_file.json")
    #处理测试集
    pretreatment("./pretreatment/SemEval/TRAIN_FILE.TXT","./dataset/sem/train_file.json")
if __name__ == '__main__':
    main()
    