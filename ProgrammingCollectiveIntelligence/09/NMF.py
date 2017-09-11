import pandas as pd
import numpy as np
from pandas import DataFrame,Series

def initialWH(v,r=5):
    w=np.random.random(v.shape[0]*r).reshape(v.shape[0],r)
    h=np.random.random(v.shape[1]*r).reshape(r,v.shape[1])
    return w,h


#Multiplicative Updata Method
def methodMUM(v,w,h):
    new_w=w*np.dot(v,h.T)/np.dot(np.dot(w,h),h.T)
    new_h=h*np.dot(new_w.T,v)/np.dot(np.dot(new_w.T,new_w),h)
    return new_w,new_h

#Gradient Approaches
def methodGA(v,w,h,alpha=0.01):
    new_w=w-alpha*(np.dot(np.dot(w,h),h.T)-np.dot(v,h.T))
    new_w[new_w<0]=0
    new_h=h-alpha*(np.dot(np.dot(w.T,w),h)-np.dot(w.T,v))
    new_h[new_h<0]=0
    return new_w,new_h

#Alternating Non-negative Least Squares
def methodALS(v,w,h):
    new_w=np.dot(np.dot(v,h.T),np.linalg.inv(np.dot(h,h.T)))
    new_w[new_w<0]=0
    new_h=np.linalg.inv(np.dot(new_w.T,new_w)).dot(new_w.T).dot(v)
    new_h[new_h<0]=0
    return new_w,new_h


#均方根误差
def RMSE(v,w,h):
    err=math.sqrt(pow(v-np.dot(w,h),2).sum()/(v.shape[0]*v.shape[1]))
    return err

#非负矩阵分解
def NMF(v,r=5,method=methodMUM,thre=0):
    w,h=initialWH(v,r)
    old_err=RMSE(v,w,h)
    while True:
        w,h=method(v,w,h)
        new_err=RMSE(v,w,h)
        if old_err-new_err<=thre:
            break
        else:
            old_err=new_err
    return w,h


if __name__=='__main__':
    item=['m'+str(i) for i in range(10)]
    user=['id'+str(i) for i in range(15)]
    rating=np.array([[5, 5, 3, 0, 5, 5, 4, 3, 2, 1, 4, 1, 3, 4, 5],
                     [5, 0, 4, 0, 4, 4, 3, 2, 1, 2, 4, 4, 3, 4, 0],
                     [0, 3, 0, 5, 4, 5, 0, 4, 4, 5, 3, 0, 0, 0, 0],
                     [5, 4, 3, 3, 5, 5, 0, 1, 1, 3, 4, 5, 0, 2, 4],
                     [5, 4, 3, 3, 5, 5, 3, 3, 3, 4, 5, 0, 5, 2, 4],
                     [5, 4, 2, 2, 0, 5, 3, 3, 3, 4, 4, 4, 5, 2, 5],
                     [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0],
                     [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                     [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                     [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]).T

    w,h=NMF(rating)
