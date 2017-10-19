import pandas as pd
import numpy as np
from pandas import DataFrame,Series

#创建基本集合
def createC1(data):
    c1=[]
    for i in range(len(data)):
        c1.extend(data[i])
    c1=set(c1)
    return map(lambda x:frozenset([x]),c1)

#创建超集
def createCk(ci):
    k=len(ci[0])+1
    ck=[]
    n=len(ci)
    for i in range(n):
        for j in range(i+1,n):
            l1=list(ci[i])[:k-2]
            l2=list(ci[j])[:k-2]
            if l1==l2:
                ck.append(ci[i] | ci[j])
    return ck

#按支持度筛选频繁集
def scanD(data,Ck,minSupport):
    supports={}
    n=len(data)
    for i in Ck:
        temp=sum(map(lambda x:i.issubset(x),data))/float(n)
        if temp>=minSupport: supports[i]=temp
    result=[k for k in supports]
    return result,supports

#Apriori
def apriori(data,minSupport=0.5):
    c1=createC1(data)
    ci,supports=scanD(data,c1,minSupport)
    L=list([ci])
    while True:
        ck=createCk(ci)
        if len(ck)==0: break
        ci,sp=scanD(data,ck,minSupport)
        if len(ci)==0: break
        L.append(ci)
        supports.update(sp)

    return L,supports

#按置信度筛选规则
def calcConf(freqSet,H,supports,minConf):
    confs=[]
    for h in H:
        if supports.get(freqSet-h,0)==0: continue
        conf=supports[freqSet]/float(supports[freqSet-h])
        if conf>=minConf:
            confs.append((freqSet-h,h,conf))
    return confs

#针对一个频繁集生成一组规则
def setRule(freqSet,H,supports,minConf):
    if len(H)==0:
        H=[frozenset([i]) for i in freqSet]
    confs=calcConf(freqSet,H,supports,minConf)
    ci=[i[1] for i in confs]
    if len(ci)>1 and len(ci[0])<len(freqSet)-1:
        H2=createCk(ci)
        return confs+setRule(freqSet,H2,supports,minConf)
    else:
        return confs


#生成规则集
def generateRule(L,supports,minConf=0.5):
    confs=[]
    for i in range(1,len(L)):
        for freqSet in L[i]:
            confs.extend(setRule(freqSet,[],supports,minConf))
    return confs


#读取国会投票数据
def readBillData(filename):
    fr=open(filename)
    data=[]
    temp=[]
    for line in fr.readlines():
        word=line.strip()
        if word.find('(')>=0 or word.find('aa')>=0:
            if len(temp)>0:
                data.append(temp)
                temp=[]
            continue
        num=int(word[(word.find('I')+1):])
        temp.append(num)
    return data


if __name__=='__main__':
    set1=[[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    L,supports=apriori(set1,minSupport=0.5)
    generateRule(L,supports,minConf=0.7)

    #国会投票数据
    filepath='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch11\\'
    data=readBillData(filepath+'bills20DataSet.txt')
    L,supports=apriori(data,0.3)
    generateRule(L,supports,0.95)

    #发现毒蘑菇相似特征
    mushroom=[line.split() for line in open(filepath+'mushroom.dat').readlines()]
    L,supports=apriori(mushroom,0.3)
    for item in L[1]:
        if item.intersection('2'):print item

