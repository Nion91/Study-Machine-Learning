import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import os


def kernelLinear(x1,x2):
    return np.dot(x1,x2)

def kernelRBF(x1,x2):
    para=-((x1-x2)**2).sum()/(2*sigma**2)
    return np.power(np.e,para)


class optStruct:
    def __init__(self,x,y,b,C,kernel=kernelLinear):
        self.x=x
        self.y=y
        self.alpha=np.zeros(x.shape[0])
        self.b=b
        self.C=C
        self.kernel=kernelLinear
        self.innermat=innerMatrix(x,kernel)
        self.ids=np.arange(x.shape[0])
        self.g=np.apply_along_axis(lambda i:svmFunc(self,i[0]),1,self.ids.reshape((x.shape[0],1)))
        self.E=self.g-self.y

    def predict(self,test):
        test=test.reshape((1,len(test))) if test.ndim==1 else test
        result=np.apply_along_axis(lambda i:svmFunc(self,i),1,test)
        return np.sign(result)



def innerMatrix(x,kernel=kernelLinear):
    mat=np.apply_along_axis(lambda i:np.apply_along_axis(lambda j:kernel(i,j),1,x),1,x)
    return mat

def svmFunc(op,x):
    k=op.innermat[x] if isinstance(x,(int,np.int64)) else np.apply_along_axis(lambda i:op.kernel(i,x),1,op.x)
    result=(op.y*op.alpha*k).sum()+op.b
    return result

def SMO(op,iters=100,thre=0.0001):
    op=optStruct(op.x,op.y,op.b,op.C,op.kernel)
    exclude_id=np.array([])
    while iters:
        #外层循环，挑选第一个变量
        first_id=None
        #循环支持向量
        outer1=op.ids[(op.alpha<op.C)&(op.alpha>0)]
        outer1=outer1[~np.in1d(outer1,exclude_id)]
        if outer1.shape[0]>0:
            ktt1=op.y[outer1]*op.g[outer1]
            if not (ktt1==1).all():
                first_id=outer1[np.abs(ktt1-1).argmax()]
        #循环非支持向量
        if first_id==None:
            outer2=op.ids[(op.alpha==0)|(op.alpha==C)]
            outer2=outer2[~np.in1d(outer2,exclude_id)]
            if outer2.shape[0]>0:
                ktt2=op.y[outer2]*op.g[outer2]
                if not np.where(outer2==0,ktt2>=1,ktt2<=1).all():
                    first_id=outer2[np.where(outer2==0,1-ktt2,ktt2-1).argmax()]
        if first_id==None: break

        #内层循环，挑选第二个变量
        E1=op.E[first_id]
        E2=op.E[op.ids!=first_id].min() if E1>0 else op.E[op.ids!=first_id].max()
        second_id=op.ids[(op.E==E2)&(op.ids!=first_id)][0]

        #更新第二个变量
        new2=updateSecondVar(op,first_id,second_id)

        #alpha2没有足够变化时重新选择alpha2
        old2=op.alpha[second_id]
        if abs(new2-old2)<=thre:
            try:
                #遍历支持向量
                if outer1[outer1!=first_id].shape[0]>0:
                    for i in outer1[outer1!=first_id]:
                        temp_new2=updateSecondVar(op,first_id,i)
                        if abs(temp_new2-op.alpha[i])>thre:
                            new2=temp_new2
                            second_id=i
                            raise new2Error
                #遍历非支持向量
                if outer2[outer2!=first_id].shape[0]>0:
                    for i in outer2[outer2!=first_id]:
                        temp_new2=updateSecondVar(op,first_id,i)
                        if abs(temp_new2-op.alpha[i])>thre:
                            new2=temp_new2
                            second_id=i
                            raise new2Error
                #重新选择alpha1            
                exclude_id=np.r_[exclude_id,first_id]
                continue
            except:
                pass

        #更新第一个变量
        new1=updateFirstVar(op,first_id,second_id,new2)

        #更新b
        op.b=updateB(op,first_id,second_id,new1,new2)
        #更新alpha
        op.alpha[first_id]=new1
        op.alpha[second_id]=new2
        #更新E
        op.g=np.apply_along_axis(lambda i:svmFunc(op,i[0]),1,op.ids.reshape((op.x.shape[0],1)))
        op.E=op.g-op.y

        exclude_id=np.array([])
        iters-=1
    return op



def updateSecondVar(op,id1,id2):
    y1=op.y[id1];y2=op.y[id2]
    old1=op.alpha[id1];old2=op.alpha[id2]
    
    if y1!=y2:
        L=max(0,old2-old1);H=min(op.C,op.C+old2-old1)
    else:
        L=max(0,old2+old1-op.C);H=min(op.C,old2+old1)

    new_unc2=old2+y2*(op.E[id1]-op.E[id2])/(op.innermat[id1,id1]+op.innermat[id2,id2]-2*op.innermat[id1,id2])
    new2=L if new_unc2<L else new_unc2 if new_unc2<=H else H
    return new2

def updateFirstVar(op,id1,id2,new2):
    y1=op.y[id1];y2=op.y[id2]
    old1=op.alpha[id1];old2=op.alpha[id2]

    new1=old1+y1*y2*(old2-new2)
    return new1


def updateB(op,id1,id2,new1,new2):
    E1=op.E[id1];E2=op.E[id2]
    y1=op.y[id1];y2=op.y[id2]
    old1=op.alpha[id1];old2=op.alpha[id2]
    
    newB1=-E1-y1*op.innermat[id1,id1]*(new1-old1)-y2*op.innermat[id2,id1]*(new2-old2)+op.b   
    newB2=-E2-y1*op.innermat[id1,id2]*(new1-old1)-y2*op.innermat[id2,id2]*(new2-old2)+op.b

    if (new1<op.C) and (new1>0): b=newB1
    elif  (new2<op.C) and (new2>0): b=newB2
    else: b=(newB1+newB2)/2.0

    return b

#自定义异常，用于跳出循环
class new2Error(Exception):
    def __init__(self):
        pass


#寻找最优sigma
def searchSigma(train,test):
    global sigma

    x=np.array(train.iloc[:,:-1])
    y=np.array(train.iloc[:,-1])
    b=0.0
    C=0.5
    op=optStruct(x,y,b,C,kernel=kernelRBF)

    test_x=np.array(test.iloc[:,:-1])
    test_y=np.array(test.iloc[:,-1])

    sig=[]
    rate=[]
    qty=[]
    for i in range(10,50,1):
        sigma=i*0.01
        newop=SMO(op)
        predict=newop.predict(test_x)
        accuracy=(predict==test_y).sum()*1.0/len(test_y)
        sig.append(sigma)
        rate.append(accuracy)
        qty.append((newop.alpha>0).sum())
    return DataFrame({'sigma':sig,'accuracy':rate,'support_qty':qty}).loc[:,['sigma','accuracy','support_qty']]



#读取手写数字数据
def getDigitsData(path):
    files=os.listdir(path)
    xlist=[]
    ylist=[]
    for f in files:
        temp=pd.read_fwf(path+os.sep+f,widths=[1]*32,names=range(32))
        xlist.append(np.array(temp).reshape(1024))
        label=1 if f[0]=='9' else -1
        ylist.append(label)
    x=np.array(xlist)
    y=np.array(ylist)
    return x,y


if __name__=='__main__':
    path='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch06\\'
    #线性可分
    data=pd.read_table(path+'testSet.txt',names=['X1','X2','Y'])
    x=np.array(data.iloc[:,:-1])
    y=np.array(data.iloc[:,-1])
    b=0.0
    C=0.5

    op=optStruct(x,y,b,C)
    op1=SMO(op)

    fig,ax=plt.subplots(1,1)
    ax.scatter(data.X1,data.X2)
    support=x[op1.alpha>0]
    ax.scatter(support[:,0],support[:,1])

    #测试RBF
    data=pd.read_table(path+'testSetRBF.txt',names=['X1','X2','Y'])
    test=pd.read_table(path+'testSetRBF2.txt',names=['X1','X2','Y'])
    x=np.array(data.iloc[:,:-1])
    y=np.array(data.iloc[:,-1])
    b=0.0
    C=0.5

    fig,ax=plt.subplots(1,1)
    ax.scatter(data.X1[y==1],data.X2[y==1])
    ax.scatter(data.X1[y==-1],data.X2[y==-1],c='red')

    op=optStruct(x,y,b,C,kernel=kernelRBF)
    
    sigma=0.4
    op1=SMO(op)

    fig,ax=plt.subplots(1,1)
    ax.scatter(data.X1[y==1],data.X2[y==1])
    ax.scatter(data.X1[y==-1],data.X2[y==-1])
    support=x[op1.alpha>0]
    ax.scatter(support[:,0],support[:,1],c='grey')

    re1=searchSigma(data,test)


    #手写数字识别
    path='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch02\\'
    train_x,train_y=getDigitsData(path+'trainingDigits')
    test_x,test_y=getDigitsData(path+'testDigits')
    b=0.0
    C=0.5

    digit=optStruct(train_x,train_y,b,C)
    d1=SMO(digit)

    predict=d1.predict(test_x)


