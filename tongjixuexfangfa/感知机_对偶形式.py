# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame,Series

class perceptron:
    def train(self,x,y,eta=1):
        self.gram=np.dot(x,x.T)
        self.eta=eta
        self.a=eta*np.zeros(x.shape[0])

        while True:
            values=Series((np.dot(self.a*y,self.gram)+np.dot(self.a,y))*y)
            if values[values<=0].count()>0:
                i=values[values<=0].index.tolist()[0]
                self.a[i]=self.a[i]+1
            else:
                self.w=np.dot(self.a*y,x)
                self.b=(self.a*y).sum()
                break

    def predict(self,x):
        y=sign(np.dot(self.w,x)+self.b)
        return y


if __name__=='__main__':
    x=np.array([[3,3],[4,3],[1,1]])
    y=np.array([1,1,-1])
    pct=perceptron()
    pct.train(x,y)
    print pct.w,pct.b

