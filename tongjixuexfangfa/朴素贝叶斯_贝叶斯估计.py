# -*- coding: utf-8 -*-

import numpy as np
import pandas as ps
from pandas import DataFrame,Series

class naiveBayes:
	def __init__(self):
		self.cond_prob=list()

	def apriProb(self,y,lambdaA=1):
		self.count_y=y.groupby(y).count().sort_index()
		self.apri_prob=(self.count_y+lambdaA)/(len(y)+len(self.count_y)*lambdaA)

	def condProbX(self,x,y,lambdaB=1):
		df=DataFrame({'x':x,'y':y})
		df=df.groupby(['x','y']).apply(lambda x:x.shape[0]).unstack().sort_index(axis=1).fillna(0)
		nx=x.unique().shape[0]
		prob=(df+lambdaB)/(self.count_y+nx*lambdaB)
		return prob

	def condProb(self,x,y,lambdaB=1):
		cols=x.shape[1]
		for i in range(cols):
			prob=self.condProbX(x.iloc[:,i],y,lambdaB)
			self.cond_prob.append(prob)

	def train(self,x,y,lamb=1):
		self.apriProb(y,lamb)
		self.condProb(x,y,lamb)

	def predictX(self,x):
		df=DataFrame(self.apri_prob).T
		for i in range(x.shape[0]):
			temp=self.cond_prob[i].loc[x[i]]
			df=df.append(temp)
		result=df.prod().sort_values(ascending=False).index[0]
		return result

	def predict(self,x):
		results=[]
		for i in range(x.shape[0]):
			res=self.predictX(x.iloc[i])
			results.append(res)
		return Series(results)


if __name__=='__main__':
	x=DataFrame({'x1':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
				 'x2':['S','M','M','S','S','S','M','M','L','L','L','M','M','L','L']})
	y=Series([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

	nb=naiveBayes()
	nb.train(x,y)
	x1=DataFrame({'x1':[2,2,3],'x2':['S','M','L']})
	nb.predict(x1)
	
