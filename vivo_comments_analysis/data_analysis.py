import pandas as pd
import numpy as np
import re
from snownlp import SnowNLP
import jieba
from gensim import corpora,models

# data = pd.read_csv('vivo_comments.csv')
# print(type(data))
'''重复值处理'''
# data_null = data.drop_duplicates()
# print(data_null)
# data_null.to_csv('comments_null.csv')
# data_null_comments = data_null['contents']
# data_null_comments.to_csv('contents.txt',index=False,encoding='utf-8')
# print(len(data_null_comments))

'''过滤短句'''
# data_len = data_null_comments[data_null_comments.str.len()>4]
# print(data_len)
# data_len.to_csv('contents.txt',index=False,encoding='utf-8')

'''机械压缩去词'''
# def str_unique(raw_str,reverse=False):
#     if reverse:
#         raw_str = raw_str[::-1]
#     res_str = ''
#     for i in raw_str:
#         if i not in res_str:
#             raw_str += i
#     if reverse:
#         res_str = res_str[::-1]
#     return res_str
# ser1 = data_comments_nonull.iloc[:,0].apply(str_unique)
# data_comments_nonull1 = pd.DataFrame(ser1.apply(str_unique,reverse=True))
# print(data_comments_nonull1)


'''情感分析'''
# data = pd.read_csv('contents.txt',encoding='utf-8',header=None)
# print(data)
# # print(type(data))
# coms = []
# coms = data[0].apply(lambda x:SnowNLP(x).sentiments)
#
# data_post = data[coms>=0.5]
# data_neg = data[coms<0.5]
# print(data_post)
# print(data_neg)
# data_post.to_csv('comments_正面情感结果.txt',encoding='utf-8',header=None)
# data_neg.to_csv('comments_负面情感结果.txt',encoding='utf-8-sig',header=None)

'''去除无用符号'''
# with open('comments_正面情感结果.txt',encoding='utf-8') as fn1:
#     string_data1 = fn1.read()    # 使用read方法读取整段文本
#
# pattern = re.compile(u'\t|\n|\.|-|——|：|！|、|，|,|。|;|\)|\(|\?|"')  # 建立正则表达式匹配模式
# string_data1 = re.sub(pattern, '', string_data1)         # 将符合模式的字符串替换掉
# print(string_data1)
#
# fp = open('comments_post.txt','a',encoding='utf8')
# fp.write(string_data1 + '\n')
# fp.close()
#
# with open('comments_负面情感结果.txt',encoding='utf-8') as fn2:
#     string_data2 = fn2.read()
#
# pattern = re.compile(u'\t|\n|\.|-|——|：|！|、|，|,|。|;|\)|\(|\?|"')  # 建立正则表达式匹配模式
# string_data2 = re.sub(pattern, '', string_data2)         # 将符合模式的字符串替换掉
# print(string_data2)
#
# fp = open('comments_neg.txt','a',encoding='utf8')
# fp.write(string_data2 + '\n')
# fp.close()

'''分词'''
# data1 = pd.read_csv('comments_post.txt',encoding='utf-8',header=None)
# data2 = pd.read_csv('comments_neg.txt',encoding='utf-8',header=None)
#
# mycut = lambda s: ' '.join(jieba.cut(s))   # 自定义简单分词函数
# data1 = data1[0].apply(mycut)
# data2 = data2[0].apply(mycut)
#
# data1.to_csv('comments_post_cut.txt',index=False,header=False,encoding='utf-8')
# data2.to_csv('comments_neg_cut.txt',index=False,header=False,encoding='utf-8')
# print(data2)

'''LDA主题分析'''
post = pd.read_csv('comments_post_cut.txt',encoding='utf-8',header=None,error_bad_lines=False)
neg = pd.read_csv('comments_neg_cut.txt',encoding='utf-8',header=None,error_bad_lines=False)
stop = pd.read_csv('stoplist.txt',encoding='utf-8',header=None,sep='tipdm',engine='python')

stop = [' ',''] + list(stop[0])   # 添加空格

post[1] = post[0].apply(lambda s: s.split(' '))
post[2] = post[1].apply(lambda x: [i for i in x if i not in stop])
neg[1] = neg[0].apply(lambda s: s.split(' '))
neg[2] = neg[1].apply(lambda x: [i for i in x if i not in stop])

'''正面主题分析'''
post_dict = corpora.Dictionary(post[2])  # 建立词典
post_corpus = [post_dict.doc2bow(i) for i in post[2]]
post_lda = models.LdaModel(post_corpus, num_topics=4, id2word=post_dict)  # LDA模型训练
for i in range(3):
    print(post_lda.print_topic(i))   # 输出每个主题


print('第一个主题分析')

'''负面主题分析'''
neg_dict = corpora.Dictionary(neg[2])  # 建立词典
neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]]
neg_lda = models.LdaModel(neg_corpus, num_topics=4, id2word=neg_dict)  # LDA模型训练
for i in range(3):
    print(neg_lda.print_topic(i))   # 输出每个主题








