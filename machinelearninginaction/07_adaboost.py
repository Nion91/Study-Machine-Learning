import pandas as pd
import numpy as np
from pandas import DataFrame,Series


#单层决策树分类器
def stumpClassify(datamat,dim,thre,rule):
    if rule=='lt':
        result=np.where(datamat[:,dim]<=thre,-1,1)
    else:
        result=np.where(datamat[:,dim]>thre,-1,1)
    return result

#单层决策树生成
def buildStump(x,y,weight):
    min_err=np.inf
    for i in range(x.shape[1]):
        numsteps=10
        stepsize=(x[:,i].max()-x[:,i].min())/numsteps
        for j in range(numsteps+1):
            thre=x[:,i].min()+j*stepsize
            for r in ['lt','gt']:
                predict=stumpClassify(x,i,thre,r)
                err=(weight[predict!=y]).sum()  #加权误差率
                if err<min_err:
                    min_err=err
                    beststump={'dim':i,'thre':thre,'rule':r}
                    bestpredict=predict
    return beststump,min_err,bestpredict


#基于单层决策树的adaboost训练
def adaBoostTrain(x,y,iters=20):
    weight=np.array([1.0/len(y)]*len(y))
    stumps=[]
    agg_predict=np.zeros(len(y))

    for i in range(iters):
        stump,err,predict=buildStump(x,y,weight)
        #计算alpha
        alpha=0.5*np.log((1-err)/max(err,1e-16))
        stump['alpha']=alpha
        stumps.append(stump)
        #计算累积误差率
        agg_predict+=alpha*predict
        agg_err=float((np.sign(agg_predict)!=y).sum())/len(y)
        #print agg_err
        if agg_err==0: break
        #更新权重
        weight=weight*np.power(np.e,-alpha*y*predict)
        weight=weight/weight.sum()

    return stumps

#adaboost分类器
def adaClassify(stumps,test):
    agg_predict=np.zeros(test.shape[0])
    for i in stumps:
        predict=stumpClassify(test,i['dim'],i['thre'],i['rule'])
        agg_predict+=i['alpha']*predict
    return np.sign(agg_predict)


if __name__=='__main__':
    x = np.array([[ 1. ,  2.1],
                  [ 2. ,  1.1],
                  [ 1.3,  1. ],
                  [ 1. ,  1. ],
                  [ 2. ,  1. ]])
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0])

    #预测病马死亡率
    path='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch05\\'
    colname=['X'+str(i) for i in range(1,22)]+['Y']
    train=pd.read_table(path+'horseColicTraining.txt',names=colname)
    test=pd.read_table(path+'horseColicTest.txt',names=colname)

    train_x=np.array(train.iloc[:,:-1])
    train_y=np.array(train.iloc[:,-1])
    train_y=np.where(train_y==0,-1,1)
    test_x=np.array(test.iloc[:,:-1])
    test_y=np.array(test.iloc[:,-1])
    test_y=np.where(test_y==0,-1,1)

    stumps=adaBoostTrain(train_x,train_y,20)
    (adaClassify(stumps,test_x)==test_y).sum()*1.0/len(test_y)


