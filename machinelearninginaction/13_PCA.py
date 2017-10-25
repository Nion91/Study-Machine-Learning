import pandas as pd
import numpy as np
from pandas import DataFrame,Series

def PCA(data,k):
    data=data-data.mean(axis=0)
    covX=data.T.dot(data)/float(data.shape[0])
    eigenvalue,eigenvector=np.linalg.eig(covX)
    index=eigenvalue.argsort()[::-1][:k]
    sub_eigenvalue=eigenvalue[index]
    print sub_eigenvalue/eigenvalue.sum()
    sub_eigenvector=eigenvector[:,index]
    A=sub_eigenvector.T
    B=A.dot(data.T)

    return B


if __name__='__main__':
    filepath='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch13\\'
    test=np.array(pd.read_table(filepath+'testSet.txt',names=['x1','x2']))
    result=PCA(test,1)

    #半导体数据
    secom=pd.read_table(filepath+'secom.data',sep=' ',names=['x'+str(i) for i in range(590)])
    secom=secom.apply(lambda x:x.fillna(x.mean()),axis=0)
    secom=np.array(secom)
    result=PCA(test,1)
