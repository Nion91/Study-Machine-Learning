import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import math

#生成树
def calcEntropy(y):
    prob=y.groupby(y).apply(lambda x:1.0*len(x)/len(y))
    entropy=prob.map(lambda x:-x*math.log(x,2)).sum()
    return entropy

def infoGain(x,y):
    before=calcEntropy(y)
    after=y.groupby(x).apply(lambda i:calcEntropy(i)*len(i)/len(y)).sum()
    gain=before-after
    return gain

def selectBestFeature(x,y):
    gain=x.apply(lambda x:infoGain(x,y))
    return gain.argmax()

def createBranch(x,y):
    category=y.groupby(y).count().argmax()
    if x.shape[0]==0 or len(y.unique())==1:
        return category
    
    feature=selectBestFeature(x,y)
    node={feature:{}}
    for i in x[feature].unique():
        ix=x.loc[x[feature]==i,x.columns!=feature]
        iy=y.loc[x[feature]==i]
        node[feature][i]=createBranch(ix,iy)
    return node

def classify(tree,x):
    key=tree.keys()[0]
    value=x[key]
    sub=tree[key][value]
    if not isinstance(sub,dict):
        return sub
    else:
        return classify(sub,x)


#解析树
def getNumLeaf(tree):
    if not isinstance(tree,dict):
        return 1

    num=0
    for v in tree.values()[0].values():
        num+=getNumLeaf(v)
    return num

def getTreeDepth(tree):
    if not isinstance(tree,dict):
        return 1
    
    temp=0
    for v in tree.values()[0].values():
        temp=max(temp,getTreeDepth(v))
    return temp+1


#plot
def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
    va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()


if __name__=='__main__':
    x=DataFrame({'a':[1,1,1,0,0],'b':[1,1,0,1,1]})
    y=Series(['yes','yes','no','no','no'],name='label')

    path='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch03\\lenses.txt'
    lenses=pd.read_table(path,names=['age','prescript','astigmatic','tearrate','label'])
    x=lenses.iloc[:,:-1]
    y=lenses.iloc[:,-1]
