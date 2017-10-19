import pandas as pd
import numpy as np
from pandas import DataFrame,Series

def loadData(filename):
    data=pd.read_table(filename)
    colnames=['x'+str(i) for i in range(data.shape[1]-1)]
    colnames.append('y')
    data=pd.read_table(filename,names=colnames)
    return np.array(data)

#子节点结构
class NodeTree:
    def __init__(self,feature,value,lefttree,righttree,isleaf,nodevalue):
        self.feature=feature
        self.value=value
        self.lefttree=lefttree
        self.righttree=righttree
        self.isleaf=isleaf
        self.nodevalue=nodevalue

    def to_dict(self):
        if self.lefttree==None:
            return {'feature':self.feature,
                    'value':self.value,
                    'lefttree':self.lefttree,
                    'righttree':self.righttree,
                    'isleaf':self.isleaf,
                    'nodevalue':self.nodevalue}
        else:
            return {'feature':self.feature,
                    'value':self.value,
                    'lefttree':self.lefttree.to_dict(),
                    'righttree':self.righttree.to_dict(),
                    'isleaf':self.isleaf,
                    'nodevalue':self.nodevalue}

#回归树
def regLeaf(data):
    return (data[:,-1]).mean()

def regErr(data):
    return (data[:,-1]).var()*data.shape[0]

def regTreeEval(model,point):
    return float(model)


#模型树
def LinearModel(data):
    x=data[:,:-1]
    if data.shape<=2: x=x.reshape((x.shape[0],1))
    x=np.concatenate((np.ones((x.shape[0],1)),x),1)
    y=data[:,-1]
    w=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return x,y,w

def modelLeaf(data):
    return LinearModel(data)[2]

def modelErr(data):
    x,y,w=LinearModel(data)
    pred=x.dot(w)
    return np.power(y-pred,2).sum()

def regModelEval(model,point):
    point=np.concatenate((np.ones(1),point))
    return point.dot(model)


#切割数据
def splitData(data,feature,value):
    d1=data[data[:,feature]>value]
    d2=data[data[:,feature]<=value]
    return d1,d2


#选择最优变量和切分值
def selectBestFeature(data,selecedFeature=[],leafType=regLeaf,errType=regErr,minImprove=1,minSample=4):
    err=np.Inf
    featureList=[i for i in range(data.shape[1]-1) if i not in selecedFeature]
    if len(featureList)==0: return None,None,leafType(data)

    d1_len=d2_len=0
    for i in featureList:
        for v in set(data[:,i]):
            d1,d2=splitData(data,i,v)
            if (d1.shape[0]<=minSample) or (d2.shape[0]<=minSample): continue
            temp_err=errType(d1)+errType(d2)
            if temp_err<err:
                err=temp_err
                feature=i
                value=v
                d1_len=d1.shape[0]
                d2_len=d2.shape[0]

    if (errType(data)-err<=minImprove) or (d1_len==0): 
        return None,None,leafType(data)
    else:
        return feature,value,leafType(data)


#生成树
def createTree(data,selecedFeature=[],leafType=regLeaf,errType=regErr,minImprove=1,minSample=4):
    feature,value,nodevalue=selectBestFeature(data,selecedFeature,leafType,errType,minImprove,minSample)
    if feature==None:
        return NodeTree(None,None,None,None,1,nodevalue)

    d1,d2=splitData(data,feature,value)
    lefttree=createTree(d1,selecedFeature+[feature],leafType,errType,minImprove,minSample)
    righttree=createTree(d2,selecedFeature+[feature],leafType,errType,minImprove,minSample)
    return NodeTree(feature,value,lefttree,righttree,0,nodevalue)


#预测
def treePredOne(tree,point,modelEval=regTreeEval):
    if tree.isleaf:
        return modelEval(tree.nodevalue,point)
    else:
        if point[tree.feature]>tree.value:
            return treePredOne(tree.lefttree,point,modelEval)
        else:
            return treePredOne(tree.righttree,point,modelEval)

def treePred(tree,points,modelEval=regTreeEval):
    if points.ndim==1: points=points.reshape((points.shape[0],1))
    pred=np.apply_along_axis(lambda x:treePredOne(tree,x,modelEval),1,points)
    return pred



if __name__=='__main__':
    filepath='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch09\\'

    #回归树
    train=loadData(filepath+'bikeSpeedVsIq_train.txt')
    test=loadData(filepath+'bikeSpeedVsIq_test.txt')
    tree=createTree(data)
    pred=treePred(tree,test[:,:-1])

    #模型树
    data=loadData(filepath+'exp2.txt')
    tree=createTree(data,leafType=modelLeaf,errType=modelErr)


