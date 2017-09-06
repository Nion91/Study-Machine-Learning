import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import math
import pprint
import copy


class DTnode:
    def __init__(self,col=None,value=None,predict=None,y=None,left=None,right=None):
        self.col=col
        self.value=value
        self.predict=predict
        self.y=y
        self.left=left
        self.right=right

#基尼系数
def giniScore(y):
    score=y.groupby(y).apply(lambda x:pow(1.0*x.count()/len(y),2)).sum()
    return 1-score

#熵
def entropyScore(y):
    score=y.groupby(y).apply(lambda x:1.0*x.count()/len(y)).map(lambda p:-p*math.log(p)).sum()
    return score

#方差
def varScore(y):
    score=y.var()
    return y

#信息增益
def infoGain(x,value,y,scorefunc=entropyScore):
    origin=scorefunc(y)
    if isinstance(value,(float,int)):
        temp=y.groupby(x<=value)
    else:
        temp=y.groupby(x==value)
    now=temp.apply(lambda y:scorefunc(y))*temp.apply(lambda x:1.0*len(x)/len(y))
    gain=origin-now.sum()
    return gain

#选取特征中最优值
def selectBestValue(x,y,scorefunc=entropyScore):
    select=x.groupby(x).apply(lambda k:infoGain(x,k.iloc[0],y,scorefunc))
    value=select.argmax()
    score=select.max()
    return Series([value,score],index=['value','score'])

#选取最优特征
def selectBestFeature(x,y,scorefunc=entropyScore):
    select=x.apply(lambda x:selectBestValue(x,y,scorefunc)).T.sort_values('score',ascending=False)
    col=select.index[0]
    value=select.value.iloc[0]
    score=select.score.iloc[0]
    return col,value,score

#分割数据
def splitData(x,y,col,value):
    if isinstance(x,Series):
        x=DataFrame(x)

    if isinstance(value,(float,int)):
        split_index=x[col]<=value
    else:
        split_index=x[col]==value

    left_x=x.loc[split_index,x.columns!=col]
    right_x=x.loc[~split_index,x.columns!=col]
    left_y=y[split_index]
    right_y=y[~split_index]

    return left_x,right_x,left_y,right_y

#CART决策树
def CART(x,y,threshold=0,scorefunc=entropyScore):
    predict=y.groupby(y).count().argmax()

    if x.shape[1]==0 or len(y.unique())==1:
        return DTnode(predict=predict,y=y)

    col,value,score=selectBestFeature(x,y,scorefunc)

    if score<=threshold:
        return DTnode(predict=predict,y=y)

    left_x,right_x,left_y,right_y=splitData(x,y,col,value)
    node=DTnode(col,value,predict,y=y,left=CART(left_x,left_y,threshold,scorefunc),right=CART(right_x,right_y,threshold,scorefunc))
    return node

#剪枝
def pruningTree(tree,threshold=0,scorefunc=entropyScore):
    node=copy.deepcopy(tree)
    if node.left==None:
        return node
    else:
        node.left=pruningTree(node.left,threshold,scorefunc)
        node.right=pruningTree(node.right,threshold,scorefunc)

    if node.left.left==None and node.right.left==None:
        before=scorefunc(node.y)
        after=scorefunc(node.left.y)*len(node.left.y)/len(node.y)+scorefunc(node.right.y)*len(node.right.y)/len(node.y)
        if before-after<=threshold:
            node.left=node.right=None
            node.col=node.value=None
    return node

#预测
def predict(x,node):
    while True:
        if node.left==None:
            break
        else:
            if isinstance(node.value,(int,float)):
                node=node.left if x[node.col]<=node.value else node.right
            else:
                node=node.left if x[node.col]==node.value else node.right
    return node.predict


#转为字典
def transNode(node):
    if node.left==None:
        dict_node={'col':node.col,'value':node.value,'predict':node.predict,'left':node.left,'right':node.right}
    else:
        dict_node={'col':node.col,'value':node.value,'predict':node.predict,'left':transNode(node.left),'right':transNode(node.right)}
    return dict_node

#打印节点
def printNode(node):
    dict_node=transNode(node)
    pprint.pprint(dict_node)



if __name__=='__main__':
    my_data=[['slashdot','USA','yes',18,'None'],
             ['google','France','yes',23,'Premium'],
             ['digg','USA','yes',24,'Basic'],
             ['kiwitobes','France','yes',23,'Basic'],
             ['google','UK','no',21,'Premium'],
             ['(direct)','New Zealand','no',12,'None'],
             ['(direct)','UK','no',21,'Basic'],
             ['google','USA','no',24,'Premium'],
             ['slashdot','France','yes',19,'None'],
             ['digg','USA','no',18,'None'],
             ['google','UK','no',18,'None'],
             ['kiwitobes','UK','no',19,'None'],
             ['digg','New Zealand','yes',12,'Basic'],
             ['slashdot','UK','no',21,'None'],
             ['google','UK','yes',18,'Basic'],
             ['kiwitobes','France','yes',19,'Basic']]

    data=[Series(i,index=['web','loc','FAQ','pages','service']) for i in my_data]
    data=pd.concat(data,axis=1).T
    x=data[['web','loc','FAQ','pages']]
    y=data.service

    tree=CART(x,y)
    test=Series(['(direct)','USA','yes',5],index=['web','loc','FAQ','pages'])
    predict(test,tree)

