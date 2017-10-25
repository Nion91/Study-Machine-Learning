import pandas as pd
import numpy as np
from pandas import DataFrame,Series

#树节点
class TreeNode:
    def __init__(self,name,count,parentnode):
        self.name=name
        self.count=count
        self.parent=parentnode
        self.children={}
        self.nodelink=None

    def increase(self,num):
        self.count+=num

    def display(self,indent=1):
        print '  '*indent,self.name,' ',self.count
        for child in self.children.values():
            child.display(indent+1)

    def displayNode(self):
        self.display()
        if self.nodelink!=None:
            self.nodelink.displayNode()

#=====================构建FPtree========================
#创建FPtree
def createFPtree(data,minSup=1):
    headtable=createHeadTable(data,minSup)
    if len(headtable)==0: return None,None

    FPtree=TreeNode('Null Set',1,None)
    for key,value in data.items():
        subset=reorderSet(key,headtable)
        if len(subset)>0:
            updateFPtree(FPtree,headtable,subset,value)
    return FPtree,headtable


#创建头指针表
def createHeadTable(data,minSup):
    headtable={}
    for subset in data:
        for item in subset:
            headtable[item]=headtable.get(item,0)+data[subset]

    for k in headtable.keys():
        if headtable[k]<minSup:
             del headtable[k]
        else:
            headtable[k]=[headtable[k],None,None]
    return headtable


#根据headtable重排序原数据集
def reorderSet(subset,headtable):
    selected=[i for i in subset if i in headtable]
    reordered=sorted(selected,key=lambda x:headtable[x][0],reverse=True)
    return reordered


#更新FPtree
def updateFPtree(FPtree,headtable,subset,count):
    first=subset[0]
    if first in FPtree.children:
        FPtree.children[first].increase(count)
    else:
        FPtree.children[first]=TreeNode(first,count,FPtree)
        updateNodeLink(headtable,FPtree.children[first],first)

    if len(subset)>1:
        updateFPtree(FPtree.children[first],headtable,subset[1:],count)


#更新节点链接
def updateNodeLink(headtable,node,item):
    if headtable[item][1]==None:
        headtable[item][1]=headtable[item][2]=node
    else:
        headtable[item][2].nodelink=node
        headtable[item][2]=node


#=====================寻找频繁项集========================
def mineTree(FPtree,headtable,minSup,prefix=set([]),freqlist=[]):
    items=[i[0] for i in sorted(headtable.items(),key=lambda x:x[0])]
    for item in items:
        newpre=prefix.copy()
        newpre.add(item)
        freqlist.append(newpre)
        subset=findPrefixPath(headtable,item)
        condFPtree,condHeadtable=createFPtree(subset,minSup)
        if condHeadtable!=None:
            mineTree(condFPtree,condHeadtable,minSup,newpre,freqlist)


#查找元素所有路径
def findPrefixPath(headtable,item):
    leafnode=headtable[item][1]
    result={}
    while leafnode!=None:
        prepath=[]
        ascendTree(leafnode,prepath)
        if len(prepath)>1:
            result[frozenset(prepath[1:])]=leafnode.count
        leafnode=leafnode.nodelink
    return result


#根据叶节点查找路径
def ascendTree(leafnode,prepath):
    if leafnode.parent!=None:
        prepath.append(leafnode.name)
        ascendTree(leafnode.parent,prepath)




#=====================测试数据集========================
#原始data转为字典形态
def to_dict(data):
    retDict={}
    for i in data:
        frozeni=frozenset(i)
        retDict[frozeni]=retDict.get(frozeni,0)+1
    return retDict

if __name__=='__main__':
    data=[['r', 'z', 'h', 'j', 'p'],
          ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
          ['z'],
          ['r', 'x', 'n', 'o', 's'],
          ['y', 'r', 'x', 'z', 'q', 't', 'p'],
          ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    data=to_dict(data)
    
    tptree,headtable=createFPtree(data,3)
    freqlist=[]
    mineTree(tptree,headtable,3,set([]),freqlist)


    data2=[['f','c','a','m','p'],
           ['f','c','a','b','m'],
           ['f','b'],
           ['c','b','p'],
           ['f','c','a','m','p']]
    data2=to_dict(data2)
    createFPtree(data)[0].display()

