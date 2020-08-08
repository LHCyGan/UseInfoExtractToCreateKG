#!/usr/bin/env python
# coding: utf-8

from IPython import get_ipython

# # 利用信息抽取技术搭建知识库
# ## 项目介绍
# - 实体统一
# - 实体识别
# - 关系抽取
# - 建立分类器
# - 操作图数据库
# - 实体消歧

# In[6]:


# get_ipython().system('pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[8]:


import jieba
import jieba.posseg as pseg
import re
import datetime
import os


# In[9]:


f_v_s_path = "../data/feature_vector.npy"
is_exist_f_v = os.path.exists(f_v_s_path)

dict_entity_name_unify = {}


# ## 一. 实体统一

# 实体统一
# 实体统一做的是对同一实体具有多个名称的情况进行统一，将多种称谓统一到一个实体上，并体现在实体的属性中（可以给实体建立“别称”属性）
# 
# 例如：对“河北银行股份有限公司”、“河北银行公司”和“河北银行”我们都可以认为是一个实体，我们就可以将通过提取前两个称谓的主要内容，得到“河北银行”这个实体关键信息。
# 
# 公司名称有其特点，例如后缀可以省略、上市公司的地名可以省略等等。在data/dict目录中提供了几个词典，可供实体统一使用。
# 
# - company_suffix.txt是公司的通用后缀词典
# - company_business_scope.txt是公司经营范围常用词典
# - co_Province_Dim.txt是省份词典
# - co_City_Dim.txt是城市词典
# - stopwords.txt是可供参考的停用词
# 

# In[10]:


# TODO：实现公司名称中地名提前
def city_prov_ahead(seg, d_city_province):
    city_prov_lst = []
    # TODO ...
    for word in seg:
        if word in d_city_province:
            city_prov_lst.append(word)
    seg_lst = [word for word in seg if word not in city_prov_lst]
    return city_prov_lst + seg_lst

# TODO：替换特殊符号
def remove_word(seg, stop_word, d_4_delete):
    filtered_word_lst = [word for word, _ in seg if word not in stop_word]
    seg_lst = [word for word in filtered_word_lst if word not in d_4_delete]
    return seg_lst


# In[19]:


# 初始化，加载词典
def my_initial():
    fr1 = open(r"../data/dict/co_City_Dim.txt", encoding='utf-8') #城市名
    fr2 = open(r"../data/dict/co_Province_Dim.txt", encoding='utf-8') #省份名
    fr3 = open(r"../data/dict/company_business_scope.txt", encoding='utf-8') # 公司业务范围
    fr4 = open(r"../data/dict/company_suffix.txt", encoding='utf-8') #公司后缀
    #城市名
    lines1 = fr1.readlines()
    d_4_delete = []
    d_city_province = [re.sub(r'(\r|\n)*','',line) for line in lines1] # 将换行符和tab转换成空字符串
    #省份名
    lines2 = fr2.readlines()
    l2_tmp = [re.sub(r'(\r|\n)*','',line) for line in lines2]
    d_city_province.extend(l2_tmp)
    #公司后缀
    lines3 = fr3.readlines()
    l3_tmp = [re.sub(r'(\r|\n)*','',line) for line in lines3]
    lines4 = fr4.readlines()
    l4_tmp = [re.sub(r'(\r|\n)*','',line) for line in lines4]
    d_4_delete.extend(l4_tmp)
    #get stop_word
    fr = open(r'../data/dict/stopwords.txt', encoding='utf-8')
    stop_word = fr.readlines()
    stop_word_after = [re.sub(r'(\r|\n)*','',stop_word[i]) for i in range(len(stop_word))]

    stop_word_after[-1] = stop_word[-1]

    stop_word = stop_word_after
    return d_4_delete,stop_word,d_city_province


# In[20]:


# 从输入的“公司名”中提取主体
def main_extract(input_str,stop_word,d_4_delete,d_city_province):
    # 开始分词并处理
    seg = pseg.cut(input_str)
    
    seg_lst = remove_word(seg, stop_word, d_4_delete)
    seg_lst = city_prov_ahead(seg_lst, d_city_province)
    
    result = ''.join(seg_lst)
    
    if result != input_str:
        if result not in dict_entity_name_unify:
            dict_entity_name_unify[result] = ""
        dict_entity_name_unify[result] = dict_entity_name_unify[result] + input_str
    return result


# In[21]:


# TODO：测试实体统一用例
d_4_delete,stop_word,d_city_province = my_initial()
company_name = "银行陕西股份有限公司"
company_name = main_extract(company_name,stop_word,d_4_delete,d_city_province)
# company_name = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体
print(company_name)


# ## 二、实体识别

# 有很多开源工具可以帮助我们对实体进行识别。常见的有LTP、StanfordNLP、FoolNLTK等等。
# 
# 本次采用FoolNLTK实现实体识别，fool是一个基于bi-lstm+CRF算法开发的深度学习开源NLP工具，包括了分词、实体识别等功能，大家可以通过fool很好地体会深度学习在该任务上的优缺点。
# 
# 在‘data/train_data.csv’和‘data/test_data.csv’中是从网络上爬虫得到的上市公司公告，数据样例如下：

# In[4]:


# get_ipython().system('pip install foolnltk -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[15]:


# import fool 注意需要在tf>=1.3and tf<=2.0
import pandas as pd
from copy import copy
from tqdm import tqdm, trange


# In[ ]:


train_data = pd.read_csv('../data/info_extract/train_data.csv', encoding = 'gb2312', header=0)
train_data.head()


# In[18]:


test_data = pd.read_csv('../data/info_extract/test_data.csv', encoding = 'gb2312', header=0)
test_data.head()


# 我们选取一部分样本进行标注，即train_data，该数据由5列组成。id列表示原始样本序号；sentence列为我们截取的一段关键信息；如果关键信息中存在两个实体之间有股权交易关系则tag列为1，否则为0；如果tag为1，则在member1和member2列会记录两个实体出现在sentence中的名称。
# 
# 剩下的样本没有标注，即test_data，该数据只有id和sentence两列，希望你能训练模型对test_data中的实体进行识别，并判断实体对之间有没有股权交易关系。
# 
# 将每句句子中实体识别出，存入实体词典，并用特殊符号替换语句。

# In[5]:


# 处理test数据，利用开源工具进行实体识别和并使用实体统一函数存储实体
import fool


# In[ ]:


test_data['ner'] = None
ner_id = 1001
ner_dict_new = {} # 存储所有实体
ner_dict_reverse_new = {} # 存储所有实体

for i in trange(len(test_data)):
    sentence = copy(test_data.iloc[i, 1])
    
    # TODO：调用fool进行实体识别，得到words和ners结果
    # TODO ...
    words, ners = fool.analysis(sentence)
    ners[0].sort(key=lambda x: x[0], reverse=True)
    for start, end, ner_type, ner_name in ners[0]:
        if ner_type == "company" or ner_type == "person":
            # TODO：调用实体统一函数，存储统一后的实体
            # 并自增ner_id
            # TODO ...
            company_main_name = main_extract(ner_name, stop_word, d_4_delete, d_city_province)
            if company_main_name not in ner_dict_new:
                ner_id += 1
                ner_dict_new[company_main_name] = ner_id
            
            # 在句子中用编号替换实体名
            sentence = sentence[:start] + ' ner_' + str(ner_dict_new[company_main_name]) + '_ ' + sentence[end - 1:]
    test_data.iloc[i, -1] = sentence
    
X_test = test_data[['ner']]


# In[ ]:


# 处理train数据，利用开源工具进行实体识别和并使用实体统一函数存储实体
train_data['ner'] = None

for i in trange(len(train_data)):
    # 判断正负样本
    if train_data.iloc[i, :]['member1'] == '0' and train_data.iloc[i, :]['member2'] == '0':
        sentence = copy(train_data.iloc[i, 1])
        # TODO：调用fool进行实体识别，得到words和ners结果
        # TODO ...
        words, ners = fool.analysis(sentence)
        ners[0].sort(key=lambda x: x[0], reverse=True)
        for start, end, ner_type, ner_name in ners[0]:
            if ner_type == 'company' or ner_type == 'person':
                # TODO：调用实体统一函数，存储统一后的实体
                # 并自增ner_id
                # TODO ...

                company_main_name = main_extract(ner_name, stop_word, d_4_delete, d_city_province)
                # company_main_name = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体

                if company_main_name not in ner_dict_new:
                    ner_id += 1
                    ner_dict_new[company_main_name] = ner_id
                    
                # 在句子中用编号替换实体名
                sentence = sentence[:start] + ' ner_' + str(ner_dict_new[company_main_name]) + '_ ' + sentence[end - 1:]
        train_data.iloc[i, -1] = sentence
    else:
        # 将训练集中正样本已经标注的实体也使用编码替换
        sentence = copy(train_data.iloc[i, :][sentence])
        for company_main_name in [train_data.iloc[i, :]['member1'], train_data.iloc[i, :]['member2']]:
            # TODO：调用实体统一函数，存储统一后的实体
            # 并自增ner_id
            # TODO ...

            company_main_name_new = main_extract(company_main_name, stop_word, d_4_delete, d_city_province)
            # company_main_name_new = ''.join(lst)  # 对公司名提取主体部分，将包含相同主体部分的公司统一为一个实体
            
            if company_main_name_new not in ner_dict_new:
                ner_id += 1
                ner_dict_new[company_main_name_new] = ner_id
            
            # 在句子中用编号替换实体名
            sentence = re.sub(company_main_name, ' ner_%s_ ' % (str(ner_dict_new[company_main_name_new])), sentence)
        train_data.iloc[i, -1] = sentence
    
ner_dict_reverse_new = {id:name for name, id in ner_dict_new.items()}

y = train_data.loc[:, ['tag']]
train_num = len(train_data)
X_train = train_data[['ner']]


# 将train和test放在一起提取特征
X = pd.concat([X_train, X_test])


# In[ ]:


# get_ipython().system('pip install pyltp -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[ ]:

"""3.关系抽取"""
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from pyltp import Segmentor

# 实体符号加入分词词典
with open("../data/user_dict.txt", 'w', encoding='utf-8') as fw:
    for v in ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']:
        fw.write(v + '号企业 ni\n')

fw.close()

# 初始化实例
segmentor = Segmentor()
# 加载模型，加载自定义词典
import os
LTP_DATA_DIR = 'G://ltp_data_v3.4.0'
# 分词模型路径，模型名称为`cws.model`
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')

# 加载模型， 加载自定义词典
segmentor.load_with_lexicon(cws_model_path, '../data/uesr_dict.txt')

# 加载停用词
fr = open(r'../data/dict/stopwords.txt', encoding='utf-8')
stop_word = fr.readlines()
stop_word = [re.sub(r'(\r|\n)*', '', stop_word[i]) for i in range(len(stop_word))]

# 分词
f = lambda x: ' '.join([word for word in segmentor.segment(re.sub(r'ner\_\d\d\d\d\_','',x)) if word not in stop_word])

corpus=X['ner'].map(f).tolist()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()  # 定一个tf-idf的vectorizer
X_tfidf = vectorizer.fit_transform(corpus).toarray()   # 结果存放在X矩阵
print(X_tfidf)

from pyltp import Parser
from pyltp import Segmentor
from pyltp import Postagger
import networkx as nx
import pylab
import re
import matplotlib.pyplot as plt
from pylab import mpl
from graphviz import Digraph
import numpy as np

# 初始化实例
postagger = Postagger()

pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
postagger.load_with_lexicon(postagger, '../data/user_dict.txt')  # 加载模型
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(cws_model_path, '../data/user_dict.txt')  # 加载模型

SEN_TAGS = ["SBV","VOB","IOB","FOB","DBL","ATT","ADV","CMP","COO","POB","LAD","RAD","IS","HED"]

def parse(s, isGraph = False):
    """
    对语句进行句法分析，并返回句法结果
    """
    tmp_ner_dict = {}
    num_lst = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']

    # print(s)

    # 将公司代码替换为特殊称谓，保证分词词性正确
    for i, ner in enumerate(list(set(re.findall(r'(ner\_\d\d\d\d\_)', s)))):
        try:
            tmp_ner_dict[num_lst[i] + '号企业'] = ner
        except IndexError:
            # TODO：定义错误情况的输出
            # TODO ...
            num_lst.append(str(i))
            tmp_ner_dict[num_lst[i] + '号企业'] = ner

        s = s.replace(ner, num_lst[i] + '号企业')

    # print(tmp_ner_dict)

    words = segmentor.segment(s)
    tags = postagger.postag(words)
    parser = Parser()  # 初始化实例

    parse_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
    parser.load(parse_model_path)

    arcs = parser.parse(words, tags)  # 句法分析
    arcs_lst = list(map(list, zip(*[[arc.head, arc.relation] for arc in arcs])))

    # 句法分析结果输出
    parse_result = pd.DataFrame([[a, b, c, d] for a, b, c, d in zip(list(words), list(tags), arcs_lst[0], arcs_lst[1])],
                                index=range(1, len(words) + 1))
    parser.release()  # 释放模型

    result = []

    # 实体的依存关系类别
    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
    # for i in range(len(words)):
    #     print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')

    company_list = list(tmp_ner_dict.keys())

    str_enti_1 = "一号企业"
    str_enti_2 = "二号企业"
    l_w = list(words)
    is_two_company = str_enti_1 in l_w and str_enti_2 in l_w
    if is_two_company:
        second_entity_index = l_w.index(str_enti_2)
        entity_sentence_type = parse_result.iloc[second_entity_index, -1]
        if entity_sentence_type in SEN_TAGS:
            result.append(SEN_TAGS.index(entity_sentence_type))
        else:
            result.append(-1)
    else:
        result.append(-1)

    if isGraph:
        g = Digraph('测试图片')
        g.node(name='Root')
        for word in words:
            g.node(name=word, fontname="SimHei")

        for i in range(len(words)):
            if relation[i] not in ['HED']:
                g.edge(words[i], heads[i], label=relation[i], fontname="SimHei")
            else:
                if heads[i] == 'Root':
                    g.edge(words[i], 'Root', label=relation[i], fontname="SimHei")
                else:
                    g.edge(heads[i], 'Root', label=relation[i], fontname="SimHei")
        g.view()

    # 企业实体间句法距离
    distance_e_jufa = 0
    if is_two_company:
        distance_e_jufa = shortest_path(parse_result, list(words), str_enti_1, str_enti_2, isGraph=False)
    result.append(distance_e_jufa)

    # 企业实体间距离
    distance_entity = 0
    if is_two_company:
        distance_entity = np.abs(l_w.index(str_enti_1) - l_w.index(str_enti_2))
    result.append(distance_entity)

    # 企业实体分别和关键触发词的距离
    key_words = ["收购", "竞拍", "转让", "扩张", "并购", "注资", "整合", "并入", "竞购", "竞买", "支付", "收购价", "收购价格", "承购", "购得", "购进",
                 "购入", "买进", "买入", "赎买", "购销", "议购", "函购", "函售", "抛售", "售卖", "销售", "转售"]
    # TODO：*根据关键词和对应句法关系提取特征（如没有思路可以不完成）
    # TODO ...

    k_w = None
    for w in words:
        if w in key_words:
            k_w = w
            break

    dis_key_e_1 = -1
    dis_key_e_2 = -1

    if k_w != None and is_two_company:
        k_w = str(k_w)
        # print("k_w", k_w)

        l_w = list(words)
        # dis_key_e_1  = shortest_path(parse_result, l_w, str_enti_1, k_w)
        # dis_key_e_2 = shortest_path(parse_result, l_w, str_enti_2, k_w)

        dis_key_e_1 = np.abs(l_w.index(str_enti_1) - l_w.index(k_w))
        dis_key_e_2 = np.abs(l_w.index(str_enti_2) - l_w.index(k_w))

    result.append(dis_key_e_1)
    result.append(dis_key_e_2)

    return result

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体

def shortest_path(arcs_ret, words, source, target, isGraph = False):
    """
    求出两个词最短依存句法路径，不存在路径返回-1
    arcs_ret：句法分析结果
    source：实体1
    target：实体2
    """
    # G = nx.DiGraph()
    G = nx.Graph()

    # 为这个网络添加节点...
    for i in list(arcs_ret.index):
        G.add_node(i)

    # TODO：在网络中添加带权中的边...（注意，我们需要的是无向边）
    # TODO ...

    for i in range(len(arcs_ret)):
        head = arcs_ret.iloc[i, -2]
        index = i + 1 # 从1开始
        G.add_edge(index, head)

    if isGraph:
        nx.draw(G, with_labels=True)
        plt.savefig("undirected_graph_2.png")
        plt.close()

    try:
        # TODO：利用nx包中shortest_path_length方法实现最短距离提取
        # TODO ...

        source_index = words.index(source) + 1 #从1开始
        target_index = words.index(target) + 1 #从1开始
        distance = nx.shortest_path_length(G, source=source_index, target=target_index)
        # print("'%s'与'%s'在依存句法分析图中的最短距离为:  %s" % (source, target, distance))

        return distance
    except:
        return -1

corpus_1 = X['ner'].tolist()
len_train_data = len(train_data)

def get_feature(s):
    """
    汇总上述函数汇总句法分析特征与TFIDF特征
    """
    # TODO：汇总上述函数汇总句法分析特征与TFIDF特征
    # TODO ...
    sen_feature = []
    len_s = len(s)
    for i in trange(len_s):
        f_e = parse(s[i], isGraph = False)
        sen_feature.append(f_e)

    sen_feature = np.array(sen_feature)

    features = np.concatenate((X_tfidf,  sen_feature), axis= 1)

    return features

features = []
if not is_exist_f_v:
    features = get_feature(corpus_1)
    np.save(f_v_s_path, features)
else:
    features = np.load(f_v_s_path)

features_train = features[:len_train_data, :]
print(features_train)

# 建立分类器进行分类
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.naive_bayes import BernoulliNB

seed = 2019

y = train_data.loc[:, ['tag']]
y = np.array(y.values)
y = y.reshape(-1)
Xtrain, Xtest, ytrain, ytest = train_test_split(features_train,  y, test_size = 0.2, random_state = seed)

def logistic_class(X_train, X_test, y_train, y_test):
    cross_validator = KFold(n_splits=10, shuffle=True, random_state=seed)

    lr = LogisticRegression(penalty='l1', solver='liblinear')

    params = {"C":[0.1,1.0,10.0,15.0,20.0,30.0,40.0,50.0]}

    grid = GridSearchCV(estimator=lr, param_grid=params, cv=cross_validator)
    grid.fit(X_train, y_train)
    print("最优参数为：", grid.best_params_)
    model = grid.best_estimator_
    y_pred = model.predict(X_test)

    y_test = [str(value) for value in y_test]
    y_pred = [str(value) for value in y_pred]

    proba_value = model.predict_proba(X_test)
    p = proba_value[:, 1]
    print("Logistic=========== ROC-AUC score: %.3f" % roc_auc_score(y_test, p))

    report = classification_report(y_pred=y_pred, y_ture=y_test)
    print(report)

    return model

# TODO：保存Test_data分类结果
# 答案提交在submit目录中，命名为info_extract_submit.csv和info_extract_entity.csv。
# info_extract_entity.csv格式为：第一列是实体编号，第二列是实体名（实体统一的多个实体名用“|”分隔）
# info_extract_submit.csv格式为：第一列是关系中实体1的编号，第二列为关系中实体2的编号。

s_model = logistic_class(Xtrain, Xtest, ytrain, ytest)

features_test = features[len_train_data:, :]
y_pred_test = s_model.predict(features_test)

l_X_test_ner = X_test.values.tolist()

entity_dict = {}
relation_list = []

for i, label in enumerate(y_pred_test):
    if label == 1:
        cur_ner_content = str(l_X_test_ner[i])

        ner_list = list(set(re.findall(r'(ner\_\d\d\d\d\_)',cur_ner_content)))
        if len(ner_list) == 2:
            # print(ner_list)
            r_e_l = []
            for i, ner in enumerate(ner_list):
                split_list = str.split(ner, "_")
                if len(split_list) == 3:
                    ner_id = int(split_list[1])

                    if ner_id in ner_dict_reverse_new:
                        if ner_id not in entity_dict:

                            company_main_name = ner_dict_reverse_new[ner_id]

                            if company_main_name in dict_entity_name_unify:
                                entity_dict[ner_id] = company_main_name + dict_entity_name_unify[company_main_name]
                            else:
                                entity_dict[ner_id] = company_main_name

                        r_e_l.append(ner_id)
            if len(r_e_l) == 2:
                relation_list.append(r_e_l)

# print(entity_dict)
# print(relation_list)
entity_list = [[item[0], item[1]] for item in entity_dict.items()]
# print(entity_list)
pd_enti = pd.DataFrame(np.array(entity_list), columns=['实体编号','实体名'])
pd_enti.to_csv("../submit/info_extract_entity.csv",index=0, encoding='utf_8_sig')

pd_re = pd.DataFrame(np.array(relation_list), columns=['实体1','实体2'])
pd_re.to_csv("../submit/info_extract_submit.csv",index=0,encoding='utf_8_sig')

from py2neo import Node, Relationship, Graph

graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    # password="person"
    password = "111111"
)

#清空所有数据对象
graph.delete_all()

for v in relation_list:
    a = Node('Company', name=str(v[0]))
    b = Node('Company', name=str(v[1]))

    # 本次不区分投资方和被投资方，无向图
    r = Relationship(a, 'INVEST', b)
    s = a | b | r
    graph.create(s)
    r = Relationship(b, 'INVEST', a)
    s = a | b | r
    graph.create(s)

# TODO：查询某节点的3层投资关系

import random

result_2 = []
result_3 = []
for value in entity_list:
    ner_id = value[0]
    str_sql_3 = "match data=(na:Company{{name:'{0}'}})-[:INVEST]->(nb:Company)-[:INVEST]->(nc:Company) where na.name <> nc.name return data".format(str(ner_id))
    result_3 = graph.run(str_sql_3).data()
    if len(result_3) > 0:
        break

if len(result_3) > 0:
    print("step1")
    print(result_3)
else:
    print("step2")
    random_index = random.randint(0, len(entity_list) - 1)
    random_ner_id = entity_list[random_index][0]
    str_sql_2 = "match data=(na:Company{{name:'{0}'}})-[*2]->(nb:Company) return data".format(str(random_ner_id))
    result_2 = graph.run(str_sql_2).data()
    print(result_2)