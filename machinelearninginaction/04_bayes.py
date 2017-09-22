import re
import string
import os
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import math

#处理数据
#读取txt
def readText(filepath):
    with open(filepath) as f:
        txt=f.read()
    return txt

#分词
def text2words(text):
    pattern=re.compile('[ %s\\b]*?(\w+)' % string.punctuation)
    words=[s.lower() for s in pattern.findall(text) if len(s)>2]
    return words

#读取folder内所有txt
def getAllText(filepath):
    words=[]
    for fp in os.listdir(filepath):
        txt=readText(filepath+fp)
        words.append(text2words(txt))
    return words

#创建词库表
def vocabList(words):
    vocab=set()
    for word in words:
        vocab=vocab.union(set(word))
    return list(vocab)

#词集模型
def word2vect(words,vocab):
    vect=np.zeros(len(words)*len(vocab)).reshape((len(words),len(vocab)))
    for i in range(len(words)):
        vect[i,:]=np.where(Series(vocab).isin(words[i]),1,0)
    return vect

#词袋模型
def bagOfWord2vect(words,vocab):
    vect=np.zeros(len(words)*len(vocab)).reshape((len(words),len(vocab)))
    for i in range(len(words)):
        temp=Series(np.zeros(len(vocab)),index=vocab)
        feq=Series(words[i]).groupby(Series(words[i])).count()
        temp.update(feq)
        vect[i,:]=np.array(temp)
    return vect


#根据idf筛选单词特征
def selectIDF(vect,k=100):
    idf=np.apply_along_axis(lambda x:math.log(vect.shape[0]/(1+x.sum())),0,vect)
    indexer=idf.argsort()[::-1][:k]
    return indexer

#分类器
def probMatrix(vect,y):
    y=Series(y)
    vect=DataFrame(vect)
    proby=y.groupby(y).apply(lambda x:float(len(x))/len(y))
    condprobx=vect.groupby(y).apply(lambda x:(1.0+x.sum())/(x.shape[0]+1))
    final=np.array(pd.concat([condprobx,proby],axis=1))
    return final

def classifyNB(probmat,vocab,x):
    indexer=word2vect([x],vocab).flatten()
    indexer=np.array(indexer.tolist()+[1])
    submat=probmat.T[indexer==1]
    result=np.apply_along_axis(lambda x:x.prod(),0,submat).argmax()
    return result

#测试模型
def testEmail(allemail,emailclass,n=1):
    accuracy=[]
    while n:
        random_index=np.random.permutation(range(len(allemail)))
        train_index=random_index[:40]
        test_index=random_index[40:]

        train_x=[allemail[i] for i in train_index]
        test_x=[allemail[i] for i in test_index]
        train_y=[emailclass[i] for i in train_index]
        test_y=[emailclass[i] for i in test_index]

        vocab=vocabList(train_x)
        vect=word2vect(train_x,vocab)
        probmat=probMatrix(vect,train_y)

        pred_y=[classifyNB(probmat,vocab,test_x[i]) for i in range(len(test_x))]
        temp_accuracy=(np.array(pred_y)==np.array(test_y)).sum()/float(len(test_y))
        accuracy.append(temp_accuracy)
        n-=1
    return np.array(accuracy).mean()


if __name__=='__main__':
    #过滤恶意留言
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]

    vocab=vocabList(postingList)
    vect=word2vect(postingList,vocab)
    probmat=probMatrix(vect,classVec)

    classifyNB(probmat,vocab,['love','my','dalmation']) #return 0
    classifyNB(probmat,vocab,['studip','garbage']) #return 1


    #过滤垃圾邮件
    filepath='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch04\\email\\'
    ham=getAllText(filepath+'ham\\')
    spam=getAllText(filepath+'spam\\')
    allemail=ham+spam
    emailclass=[0]*len(ham)+[1]*len(spam)

    testEmail(allemail,emailclass,n=10)
