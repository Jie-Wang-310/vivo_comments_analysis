'''
非结构化数据分析
作者：Jack
'''

import re
import collections     # 词频统计库
import numpy as np
import jieba
import wordcloud     # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt



'''使用此方法读取文本文件'''
with open('comments_neg.txt',encoding='utf-8') as fn:
    string_data = fn.read()    # 使用read方法读取整段文本

'''文本预处理'''

pattern = re.compile(u'\t|\n|\.|-|——|：|！|、|，|。|;|\)|\(|\?|"')  # 建立正则表达式匹配模式
string_data = re.sub(pattern, '', string_data)         # 将符合模式的字符串替换掉
print(string_data)

'''文本分词'''

seg_list_exact = jieba.cut(string_data, cut_all=False)     # 精确模式分词
remove_words = [u'的', u'，', u'和', u'是', u'随着', u'对于', ' ', u'对', u'等', u'能', u'都', u'。',
                u'、', u'中', u'与', u'在', u'其', u'了', u'可以', u'进行', u'有', u'更', u'需要', u'提供',
                u'多', u'能力', u'通过', u'会', u'不同', u'一个', u'这个', u'我们', u'将', u'并',
                u'同时', u'看', u'如果', u'但', u'到', u'非常', u'—', u'如何', u'包括', u'这',u'！',u'很',u'：',
                u'也',u'买',u'我',u'就',u'不']
# 自定义去除词,定义去除词以后即这些词不参与词频统计，使结果更有效

# remove_words = []  # 空的去除词列表，就是不去除词，用于跟关键字提取做对比
object_list = [i for i in seg_list_exact if i not in remove_words]# 将不在去除词列表中的词添加到列表中
print(object_list)
object_list = str(object_list)
fp = open('comments_neg_cut.txt','a',encoding='utf8')
fp.write(object_list + '\n')
fp.close()



'''词频统计'''

# word_counts = collections.Counter(object_list)   # 对分词作词频统计
# word_counts_top5 = word_counts.most_common(50)    # 获取前5个频率最高的词
# for w, c in word_counts_top5:            # 分别读出每条词和出现次数
#     print(w, c)
#
# '''词频展示'''
#
# mask = np.array(Image.open('wordcloud.jpg'))       # 定义词频背景，放一张背景图到文件夹中并命名
# wc = wordcloud.WordCloud(
#     font_path='C://Windows//Fonts//simhei.ttf',    # 设置字体格式，不设置将无法显示中文  字体就是所示文件夹中
#     mask=mask,                                     # 设置背景图
#     max_words=200,                                 # 设置最大展示的词数
#     max_font_size=100                              # 设置字体最大值
# )
# wc.generate_from_frequencies(word_counts)          # 从字典生成词云
# image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
# wc.recolor(color_func=image_colors)                # 将词云颜色设置为背景图方案
# pic_output = 'wordcloud'
# plt.imshow(wc)                                     # 显示词云
#
# # plt.show()
# plt.axis('off')                                    # 关闭坐标轴
# plt.show()
# plt.savefig(u'%s.png' %(pic_output),dpi=500)
#
#
#
#
#
#
#



