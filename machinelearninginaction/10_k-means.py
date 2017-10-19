import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import math


def loadData(filename):
    data=pd.read_table(filename)
    colnames=['x'+str(i) for i in range(data.shape[1]-1)]
    colnames.append('y')
    data=pd.read_table(filename,names=colnames)
    return np.array(data)


#距离函数
def distEuclidean(x,y):
    return math.sqrt(np.power(x-y,2).sum())


#初始化质心
def initCenter(data,k):
    return np.apply_along_axis(lambda x:np.random.uniform(x.min(),x.max(),k),0,data)

#k-mean
def kMeans(data,k,distFunc=distEuclidean,createCenter=initCenter):
    centers=createCenter(data,k)
    groups=np.zeros(data.shape[0])
    while True:
        distances=np.apply_along_axis(lambda x:np.apply_along_axis(lambda y:distFunc(x,y),1,centers),1,data)
        temp_groups=np.apply_along_axis(lambda x:x.argmin(),1,distances)
        #停止条件
        change_rate=(temp_groups!=groups).sum()/float(len(groups))
        if change_rate<=0: break

        groups=temp_groups
        for j in range(k):
            centers[j]=data[groups==j].mean(axis=0)

    return centers,groups

#二分k-mean
def biKmeans(data,k,distFunc=distEuclidean):
    #初始化簇
    groups=np.zeros(data.shape[0])
    centerlist=[data.mean(axis=0)]

    while len(centerlist)<k:
        index=0
        err_improve=0
        #对每个簇进行二分，找出改善误差最大的簇
        for i in range(len(centerlist)):
            subdata=data[groups==i]
            err_before=np.power(np.apply_along_axis(lambda x:distFunc(x,centerlist[i]),1,subdata),2).sum()
            temp_centers,temp_groups=kMeans(subdata,2,distFunc)
            err_after1=np.power(np.apply_along_axis(lambda x:distFunc(x,temp_centers[0]),1,subdata[temp_groups==0]),2).sum()
            err_after2=np.power(np.apply_along_axis(lambda x:distFunc(x,temp_centers[1]),1,subdata[temp_groups==1]),2).sum()
            temp_improve=err_before-err_after1-err_after2
            if temp_improve>err_improve:
                index=i
                err_improve=temp_improve
                best_centers=temp_centers
                best_groups=temp_groups
        #更新簇和质心
        groups[groups==index]=best_groups+groups.max()+1
        groups[groups>index]=groups[groups>index]-1
        centerlist.pop(index)
        centerlist.extend([best_centers[i] for i in range(best_centers.shape[0])])

    centers=np.concatenate([i.reshape((1,data.shape[1])) for i in centerlist])
    return centers,groups





if __name__=='__main__':
    filepath='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch10\\'

    data=loadData(filepath+'testSet.txt')
    kMeans(data,4)

    data2=loadData(filepath+'testSet2.txt')
    biKmeans(data2,3)

    #clubs
    places=pd.read_table(filepath+'places.txt',sep='\t',names=['club','address','area','x','y'])
    location=np.array(places.loc[:,['x','y']])
    biKmeans(location,4)
