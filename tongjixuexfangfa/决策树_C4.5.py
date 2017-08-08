# coding: utf-8

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import math

class DecisionTree:
	#熵
	def entropy(self,y):
		prob=y.groupby(y).count()/len(y)
		ent=prob.apply(lambda x:-x*math.log(x)).sum()
		return ent
	
	#条件熵
	def condEntropy(self,x,y):
		ents=y.groupby(x).apply(lambda x:self.entropy(x))
		probs=x.groupby(x).count()/len(x)
		cond_entropy=(ents*probs).sum()
		return cond_entropy

	#信息增益
	def infoGain(self,x,y):
		entropy=self.entropy(y)
		cond_entropy=self.condEntropy(x,y)
		gain=entropy-cond_entropy
		return gain

	#信息增益率
	def infoGainRate(self,x,y):
		entropy=self.entropy(y)
		cond_entropy=self.condEntropy(x,y)
		gain_rate=(entropy-cond_entropy)/entropy
		return gain_rate

	#特征筛选
	def selectVar(self,x,y):
		gain_rates=x.apply(lambda x:self.infoGainRate(x,y))
		var=gain_rates.argmax()
		maxinfogain=gain_rates.max()
		return (var,maxinfogain)

	#生成树
	def getTree(self,x,y,threshold=0,deep=0):
		predict_y=y.groupby(y).count().argmax()
		exp_entropy=y.shape[0]*self.entropy(y)
		node={'var':None,'deep':deep,'isleaf':True,'predict':predict_y,'entropy':exp_entropy,'childtree':None}
		#无可分的变量或只剩下一个类
		if x.shape[1]==0 or len(y.unique())==1:
			return node
		#信息增益率不超过阈值
		var,infogain=self.selectVar(x,y)
		if infogain<=threshold:
			return node

		varvalue=x.loc[:,var]
		node['var']=var
		node['isleaf']=False
		node['childtree']={}
		for i in varvalue.unique():
			sub_x=x.loc[varvalue==i,x.columns!=var]
			sub_y=y.loc[varvalue==i]
			node['childtree'][i]=self.getTree(sub_x,sub_y,threshold,deep+1)

		return node
	
	#整颗树的经验熵
	def expEntropy(self,tree):
		if tree['isleaf']:
			return tree['entropy']
		else:
			total_entropy=0
			for i in tree['childtree'].values():
				total_entropy+=self.expEntropy(i)
			return total_entropy

	#剪枝
	def pruningTree(self,tree,alpha=0.5):
		leafnodes=[]
		loss_before=0
		for i in tree['childtree'].values():
			if not i['isleaf']:
				i=self.pruningTree(i,alpha)
			leafnodes.append(i['isleaf'])
			loss_before+=i['entropy']+alpha

		if all(leafnodes):
			loss_after=tree['entropy']+alpha
			if loss_after<=loss_before:
				tree['var']=None
				tree['isleaf']=True
				tree['childtree']=None

		return tree

	#预测单个case
	def predictOneCase(self,x,tree):
		if tree['isleaf']:
			return tree['predict']
		else:
			return self.predictOneCase(x,tree['childtree'].get(x[tree['var']]))



	#生成模型
	def fit(self,x,y,threshold=0,alpha=0):
		self.tree=self.getTree(x,y,threshold)
		if alpha>0:
			self.tree=self.pruningTree(self.tree,alpha)

	#预测
	def predict(self,x):
		return x.apply(self.predictOneCase,axis=1,tree=self.tree)



if '__name__'=='__main__':
	df=DataFrame({'age':['Y','Y','Y','Y','Y','M','M','M','M','M','O','O','O','O','O'],
				  'work':['F','F','T','T','F','F','F','T','F','F','F','F','T','T','F'],
				  'house':['F','F','F','T','F','F','F','T','T','T','T','T','F','F','F'],
				  'debt':['C','B','B','C','C','C','B','B','A','A','A','B','B','A','C'],
				  'label':['F','F','T','T','F','F','F','T','T','T','T','T','T','T','F']})
	x=df.loc[:,['age','work','house','debt']]
	y=df['label']

	DT=DecisionTree()
	DT.fit(x,y)
	DT.predict(x)
