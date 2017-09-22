import numpy as np
import pandas as pd
from pandas import DataFrame,Series

#辅助函数
def getNewX(x):
    x0=np.ones(x.shape[0]).reshape((x.shape[0],1))
    newx=np.concatenate((x0,x),axis=1)
    return newx

#sigmoid函数
def sigmoid(x,w):
    z=np.dot(x,w)
    return 1.0/(1+np.exp(-z))

#梯度上升
def gradientAscent(x,y,alpha=0.001,n=100):
    newx=getNewX(x)
    weight=np.ones(newx.shape[1]) #初始化权重

    while n:
        expect=np.apply_along_axis(sigmoid,1,newx,w=weight)
        err=y-expect
        weight=weight+alpha*np.apply_along_axis(lambda x:(x*err).sum(),0,newx)
        n-=1

    return weight

#随机梯度上升
def randomGradAscent(x,y,alpha=0.01,iters=1000):
    newx=getNewX(x)
    weight=np.ones(newx.shape[1])

    for i in range(iters):
        wrong_index=np.arange(newx.shape[0])[classifyLogit(x,weight)!=y]
        if len(wrong_index)==0: break
        index=np.random.choice(wrong_index) #随机选择一个错误案例
        err=y[index]-sigmoid(newx[index],weight)
        weight=weight+alpha*newx[index]*err

    return weight

#预测
def classifyLogit(x,w):
    newx=getNewX(x)
    score=np.apply_along_axis(sigmoid,1,newx,w=w)
    result=np.where(score>0.5,1,0)
    return result

#测试模型
def testHorse(train,test,alpha=0.01,iters=1000,n=10):
    accuracy=[]
    for i in range(n):
        train_x=np.array(train.iloc[:,:-1])
        train_y=np.array(train.iloc[:,-1])
        test_x=np.array(test.iloc[:,:-1])
        test_y=np.array(test.iloc[:,-1])

        #weight=gradientAscent(train_x,train_y,alpha,iters)
        weight=randomGradAscent(train_x,train_y,alpha,iters)
        pred=classifyLogit(test_x,weight)
        result=(pred==test_y).sum()/float(len(test_y))
        accuracy.append(result)
    return np.array(accuracy).mean()


if __name__=='__main__':
    path='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch05\\'
    #简单二维数据
    data=pd.read_table(path+'testSet.txt',names=['X1','X2','Y'])
    x=np.array(data.iloc[:,:-1])
    y=np.array(data.iloc[:,-1])
    w=gradientAscent(x,y,n=500)
    (classifyLogit(x,w)==y).sum()/float(len(y))


    #预测病马死亡率
    path='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch05\\'
    colname=['X'+str(i) for i in range(1,22)]+['Y']
    train=pd.read_table(path+'horseColicTraining.txt',names=colname)
    test=pd.read_table(path+'horseColicTest.txt',names=colname)

