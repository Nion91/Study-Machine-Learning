import pandas as pd
import numpy as np
from pandas import DataFrame,Series


def loadData(filename):
    data=pd.read_table(filename)
    colnames=['x'+str(i) for i in range(data.shape[1]-1)]
    colnames.append('y')
    data=pd.read_table(filename,names=colnames)
    x=np.array(data.iloc[:,:-1])
    y=np.array(data.iloc[:,-1])
    return x,y

#普通线性回归
def regOLR(points,x,y):
    w=np.linalg.inv(np.dot(x.T,x)).dot(x.T).dot(y)
    return points.dot(w)

#局部加权线性回归
def regLWLR_OneCase(testpoint,x,y,k=1.0):
    weights=np.apply_along_axis(lambda i:np.exp(np.linalg.norm(i-testpoint)**2/(-2.0*k**2)),1,x)
    xTx=x.T.dot(x*weights.reshape((weights.shape[0],1)))
    w=np.linalg.inv(xTx).dot(x.T.dot(y*weights))
    return (w*testpoint).sum()

def regLWLR(points,x,y,k=1.0):
    result=np.apply_along_axis(lambda i:regLWLR_OneCase(i,x,y,k),1,points)
    return result

#残差平方和
def rssError(pred,y):
    return power(pred-y,2).sum()

#岭回归
def regRidge(points,x,y,lam=0.2):
    penalty=np.eye(x.shape[1])*lam
    w=np.linalg.inv(x.T.dot(x)+penalty).dot(x.T).dot(y)
    return points.dot(w)

#向前逐步回归
def stageWise(x,y,step=0.01,iters=100):
    #标准化数据
    m,n=x.shape
    x=(x-x.mean(axis=0).reshape((1,n)))/x.var(axis=0).reshape((1,n))
    y=y-y.mean()

    #贪心搜索
    ws=wsmax=np.zeros(n)
    for i in range(iters):
        err=np.Inf
        for j in range(n):
            for k in [-1,1]:
                temp_ws=ws.copy()
                temp_ws[j]+=k*step
                pred=x.dot(temp_ws)
                temp_err=rssError(pred,y)
                if temp_err<err:
                    err=temp_err
                    wsmax=temp_ws
        ws=wsmax.copy()
    return ws



if __name__=='__main__':
    filepath='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch08\\'
    x1,y1=loadData(filepath+'ex0.txt')

    regLWLR(x1,x1,y1,1.0)
    regLWLR(x1,x1,y1,0.001)

    #预测鲍鱼年龄
    abX,abY=loadData(filepath+'abalone.txt')
    train_x,train_y=abX[0:99],abY[0:99]
    test_x,test_y=abX[100:199],abY[100:199]

    rssError(regOLR(test_x,train_x,train_y),test_y)
    rssError(regLWLR(test_x,train_x,train_y,10),test_y)
