import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import os


def distEuclidean(x1,x2):
    dist=((x1-x2).map(lambda x:x**2).sum())**0.5
    return dist

def knn(train,result,x,k=5):
    dist=train.apply(lambda i:distEuclidean(i,x),axis=1).sort_values()
    index=dist.index[:k]
    result=result[index].groupby(result[index]).count().argmax()
    return result

#归一化
def autoNorm(x):
    new=x.apply(lambda x:(x-x.min())/(x.max()-x.min()))
    return new

#精确率
def accuracyRate(predict,y):
    rate=1.0*(predict==y).sum()/len(y)
    return rate

#拆分数据
def splitData(x,y,rate=0.1):
    index=np.random.permutation(x.index)
    num=int(x.shape[0]*rate)
    test_index=index[:num]
    train_index=index[num:]

    test_x=x.loc[test_index]
    train_x=x.loc[train_index]
    test_y=y.loc[test_index]
    train_y=y.loc[train_index]

    return train_x,test_x,train_y,test_y

#测试
def dataSetTest(data,k=5,testrate=0.1):
    x=autoNorm(data.iloc[:,:-1])
    y=data.iloc[:,-1]

    train_x,test_x,train_y,test_y=splitData(x,y,rate=testrate)
    pred=test_x.apply(lambda x:knn(train_x,train_y,x,k=k),axis=1)
    accuracy=accuracyRate(pred,test_y)
    return accuracy

#预测
def dataSetPredict(train,result,x,k=5):
    newtrain=autoNorm(train)
    x=(x-train.min())/(train.max()-train.min())
    pred=knn(newtrain,result,x,k)
    return pred


#手写数字识别
def getDigitsData(path):
    files=os.listdir(path)
    xlist=[]
    ylist=[]
    for f in files:
        temp=pd.read_fwf(path+os.sep+f,widths=[1]*32,names=range(32))
        xlist.append(np.array(temp).reshape((1,1024)))
        ylist.append(f[0])
    x=DataFrame(np.concatenate(xlist))
    y=Series(ylist)
    return x,y



if __name__=='__main__':
    path='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch02\\'
    dateset=pd.read_table(path+'datingTestSet.txt',names=['flight','game','icecream','label'])
    dataSetTest(dateset)

    
    path='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch02\\'
    train_x,train_y=getDigitsData(path+'trainingDigits')
    test_x,test_y=getDigitsData(path+'testDigits')
    pred=test_x.apply(lambda x:knn(train_x,train_y,x,k=k),axis=1)
    accuracy=accuracyRate(pred,test_y)
