# coding: utf-8

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import math

class CART:
	#基尼指数
	def giniScore(self,y):
		probs=y.groupby(y).count()/len(y)
		gini=1-probs.pow(2).sum()
		return gini

	#按指定值分类后的基尼指数
	def dichGiniScore(self,x,y,value):
		if x.dtype in [np.int64,np.float64]:
			ginis=y.groupby(x<=value).apply(self.giniScore)*y.groupby(x<=value).count()/len(y)
		else:
			ginis=y.groupby(x==value).apply(self.giniScore)*y.groupby(x==value).count()/len(y)
		return ginis.sum()

	#选择特征最优值
	def selectValue(self,x,y):
		values=x.drop_duplicates()
		scores=values.apply(lambda v:self.dichGiniScore(x,y,v))
		score=scores.min()
		value=values[scores.argmin()]
		return value,score

	#选择最优特征
	def selectVar(self,x,y):
		var_df=x.apply(lambda x:Series(self.selectValue(x,y)))
		var=var_df.iloc[1].argmin()
		value=var_df.iloc[0][var]
		return var,value

	#拆分数据
	def splitSamples(self,x,y,var,value):
		split=x[var]
		if split.dtype in [np.int64,np.float64]:
			left_index=split<=value
			right_index=split>value
		else:
			left_index=split==value
			right_index=split!=value

		left_x=x.loc[left_index,x.columns!=var]
		left_y=y.loc[left_index]
		right_x=x.loc[right_index,x.columns!=var]
		right_y=y.loc[right_index]
		return left_x,left_y,right_x,right_y

	#生成树
	def getTree(self,x,y,n=0,threshold=0,deep=0):
		predict_y=y.groupby(y).count().argmax()
		gini_score=self.giniScore(y)*len(y)
		
		if x.shape[1]==0 or y.shape[0]<=n or gini_score<=threshold:
			node={'deep':deep,
				  'var':None,
				  'value':None,
				  'lefttree':None,
				  'righttree':None,
				  'isleaf':True,
				  'predict':predict_y,
				  'gini':gini_score}
			return node

		var,value=self.selectVar(x,y)
		left_x,left_y,right_x,right_y=self.splitSamples(x,y,var,value)
		node={'deep':deep,
			  'var':var,
			  'value':value,
			  'lefttree':self.getTree(left_x,left_y,n,threshold,deep+1),
			  'righttree':self.getTree(right_x,right_y,n,threshold,deep+1),
			  'isleaf':False,
			  'predict':predict_y,
			  'gini':gini_score}

		return node

	#计算整颗树的经验损失和子节点数
	def treeExpLoss(self,tree):
		if tree['isleaf']:
			loss=tree['gini']
			nodes_n=1
		else:
			l1,n1=self.treeExpLoss(tree['lefttree'])
			l2,n2=self.treeExpLoss(tree['righttree'])
			loss=l1+l2
			nodes_n=n1+n2
		return loss,nodes_n

	#计算可以对该节点进行剪枝时最小的alpha
	def nodeAlpha(self,tree):
		if tree['isleaf']:
			return None
		l1,n1=self.treeExpLoss(tree)
		l2,n2=tree['gini'],1
		alpha=(l2-l1)/(n1-n2)
		return alpha


	#生成所有节点的alpha及路径
	def generateAllNodes(self,tree,nodepath=[]):
		if not tree['isleaf']:
			nodes=[(self.nodeAlpha(tree),nodepath)]
			nodes.extend(self.generateAllNodes(tree['lefttree'],nodepath+['lefttree']))
			nodes.extend(self.generateAllNodes(tree['righttree'],nodepath+['righttree']))
		else:
			nodes=[]
		return nodes

	#根据节点路径剪枝
	def pruningNode(self,tree,nodepath=[]):
		if len(nodepath)==0:
			tree={'deep':tree['deep'],
				  'var':None,
				  'value':None,
				  'lefttree':None,
				  'righttree':None,
				  'isleaf':True,
				  'predict':tree['predict'],
				  'gini':tree['gini']}
		else:
			childtreepath=nodepath.pop(0)
			tree[childtreepath]=self.pruningNode(tree[childtreepath],nodepath)
		return tree

	#选择最优节点进行剪枝
	def selectNode(self,tree):
		nodes=self.generateAllNodes(tree)
		nodes.sort(key=lambda x:x[0])
		min_alpha=nodes[0][0]
		node_path=nodes[0][1]

		return min_alpha,self.pruningNode(tree,node_path)

	#生成一系列子树
	def generateSeriesTrees(self,tree,first=True):
		if tree['isleaf']:
			return []
		else:
			alpha,prunedtree=self.selectNode(tree)
			treesets=[(alpha,prunedtree)]+self.generateSeriesTrees(prunedtree,False)
			if first:
				treesets=[(0,tree)]+treesets
			return treesets

	#把测试集按树进行分组
	def splitTest(self,tree,testx,testy):
		if tree['isleaf']:
			return [testy]
		else:
			left_x,left_y,right_x,right_y=self.splitSamples(testx,testy,tree['var'],tree['value'])
			return self.splitTest(tree['lefttree'],left_x,left_y)+self.splitTest(tree['righttree'],right_x,right_y)

	#计算泛化能力
	def treeGenelization(self,tree,testx,testy):
		predicts=self.splitTest(tree,testx,testy)
		giniscores=[self.giniScore(i)*len(i) for i in predicts]
		return sum(giniscores)

	#选择最优子树
	def selectTree(self,trees,testx,testy):
		giniscores=[self.treeGenelization(tree[1],testx,testy) for tree in trees]
		index=Series(giniscores).argmin()
		return trees[index]

	#预测单个case
	def predictOneCase(self,x,tree):
		if tree['isleaf']:
			result=tree['predict']
		else:
			value_x=x[tree['var']]
			if isinstance(value_x,(int,float)):
				result=self.predictOneCase(x,tree['lefttree']) if value_x<=tree['value'] else self.predictOneCase(x,tree['righttree'])
			else:
				result=self.predictOneCase(x,tree['lefttree']) if value_x==tree['value'] else self.predictOneCase(x,tree['righttree'])
		return result


	#拟合模型
	def fit(self,trainx,trainy,testx,testy,n=0,threshold=0):
		tree=self.getTree(trainx,trainy,n,threshold)
		self.treesets=self.generateSeriesTrees(tree)
		self.tree=self.selectTree(self.treesets,testx,testy)

	#预测
	def predict(self,x):
		return x.apply(self.predictOneCase,axis=1,tree=self.tree)


if __name__=='__main__':
	df=DataFrame({'age':['Y','Y','Y','Y','Y','M','M','M','M','M','O','O','O','O','O'],
				  'work':['F','F','T','T','F','F','F','T','F','F','F','F','T','T','F'],
				  'house':['F','F','F','T','F','F','F','T','T','T','T','T','F','F','F'],
				  'debt':['C','B','B','C','C','C','B','B','A','A','A','B','B','A','C'],
				  'label':['F','F','T','T','F','F','F','T','T','T','T','T','T','T','F']})
	x=df.loc[:,['age','work','house','debt']]
	y=df['label']

	ct=CART()
	ct.getTree(x,y)
