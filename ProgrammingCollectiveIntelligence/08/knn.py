import random
import math
import pandas as pd
import numpy as np
from pandas import DataFrame,Series

#生成数据集
def winePrice(rating,age):
    peak_age=rating-50

    #根据等级计算价格
    price=rating/2.0
    price=price*np.where(age>peak_age,5-(age-peak_age),5*(age+1)/peak_age)
    price[price<0]=0

    return price

def wineSet(n=200,noise=True):
    wines=DataFrame({'rating':np.random.random(n)*50+50,'age':np.random.random(n)*50})[['rating','age']]
    wines['price']=winePrice(wines.rating,wines.age)

    #增加噪声
    if noise:
        wines['price']=wines.price*(np.random.random(n)*0.4+0.8)

    return wines

def wineSet2(n=200):
    wines=DataFrame({'rating':np.random.random(n)*50+50,
                     'age':np.random.random(n)*50,
                     'aisle':np.random.randint(1,20,n),
                     'bottlesize':np.random.choice([375.0,750.0,1500.0,3000.0],n)})[['rating','age','aisle','bottlesize']]
    wines['price']=winePrice(wines.rating,wines.age)
    wines['price']=wines.price*(wines.bottlesize/750.0)
    wines['price']=wines.price*(np.random.random(n)*0.9+0.2)
    return wines

def wineSet3(n=200):
    wines=wineSet(n)
    wines['price']=np.where(np.random.random(n)<0.5,wines.price*0.5,wines.price)
    return wines

#knn
def onecaseKnn(data,test,k=3,weightfunc=None):
    dist=distEuclidian(data.iloc[:,:-1],test).sort_values().iloc[:k]
    price=data.loc[dist.index.tolist(),'price']
    if weightfunc==None:
        result=price.mean()
    else:
        weight=weightfunc(dist)
        result=(price*weight).sum()/weight.sum()
    return result

def knn(data,test,k=3,weightfunc=None):
    price=test.apply(lambda x:onecaseKnn(data,x,k,weightfunc),axis=1)
    return price

def distEuclidian(x,y):
    dist=x.apply(lambda x:math.sqrt((x-y).pow(2).sum()),axis=1)
    return dist

#权重
#反函数
def weightAntifunc(dist,num=1,const=0.1):
    weight=num/(dist+const)
    return weight

#减法函数
def weightSubtraction(dist,const=1):
    weight=np.where(const>dist,const-dist,0)
    return weight

#高斯函数
def weightGaussian(dist,sigma=10):
    weight=math.e**(-dist.pow(2)/(2*sigma**2))
    return weight


#交叉验证
def crossValid(data,test=0.1,k=3,is_random=False):
    if is_random: data=data.iloc[np.random.permutation(range(data.shape[0]))]
    i=int(data.shape[0]*(1-test))
    trainset=data.iloc[:i]
    testset=data.iloc[i:]
    predict=knn(trainset,testset.iloc[:,:-1],k=k,weightfunc=weightGaussian)
    cost=(testset.price-predict).abs().sum()
    return cost

#数据缩放
def rescale(data,scales):
    data=data.copy()
    for i in range(len(scales)):
        data.iloc[:,i]=data.iloc[:,i]*scales[i]
    return data

#用优化算法找缩放比例
class ScaleTask:
    def __init__(self,data):
        self.data=data
        self.domain=DataFrame({'lower':0,'upper':[20]*4})

    def decodeGene(self,gene):
        return gene.tolist()

    def cost(self,gene):
        scales=self.decodeGene(gene)
        cost=crossValid(rescale(self.data,scales))
        return cost


if __name__=='__main__':
    wines=wineSet(200)
    test=wineSet(50,noise=False)
    test['predict']=knn(wines,test.iloc[:,:-1])

