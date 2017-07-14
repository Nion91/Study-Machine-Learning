# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class perceptron:
    def update(self,x,y):
        self.w=self.w+self.eta*y*x
        self.b=self.b+self.eta*y

    def train(self,x,y,eta=1):
        self.w=np.zeros((x.shape[1]))
        self.b=0
        self.eta=eta

        while True:
            wrong_sign=(np.dot(self.w,x.T)+self.b)*y<=0
            if wrong_sign.sum()>0:
                xi=x[wrong_sign][0]
                yi=y[wrong_sign][0]
                self.update(xi,yi)
            else:
                break

    def predict(slef,x):
        y=sign(np.dot(self.w,x.T)+self.b)
        return y

if __name__=='__main__':
    x=np.array([[3,3],[4,3],[1,1]])
    y=np.array([1,1,-1])
    pct=perceptron()
    pct.train(x,y)
    print pct.w,pct.b
