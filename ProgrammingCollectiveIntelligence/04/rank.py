import numpy as np
import pandas as pd
from pandas import DataFrame,Series

#归一化
def normalizeScore(scores,smallerisbetter=False):
	min_limit=0.0001
	if smallerisbetter:
		base=max(scores.min(),min_limit)
		nscores=base/scores.apply(lambda x:max(x,min_limit))
	else:
		base=max(scores.max(),min_limit)
		nscores=scores/base



###基于内容排名
'''
data.columns=['urlid','word1_loc','word2_loc',...]
'''
#单词频率
def scoreWordFreq(data):
	scores=data.groupby(data.urlid).count()
	return normalizeScore(scores)

#文档位置
def scoreWordLocation(data):
	scores=data.groupby(data.urlid).apply(lambda x:x.iloc[:,1:].sum(axis=1).min())
	return normalizeScore(scores,smallerisbetter=True)

#单词距离
def scoreWordDistance(data):
	scores=data.groupby(data.urlid).apply(lambda x:x.iloc[:,1:].apply(lambda x:x.diff(1).abs().sum(),axis=1).min())
	return normalizeScore(scores,smallerisbetter=True)



###基于外部回指链接
'''
data.columns=['from_urlid','to_urlid']
'''
#PageRank
def scorePageRank(data,iters=20,damp=0.85,minpr=0.15):
	urls=data.iloc[:,0].append(data.iloc[:,1]).drop_duplicates().tolist()
	#初始化pr
	pageranks=Series(1,index=urls,name='pr')

	links=data.iloc[:,0].groupby(data.iloc[:,0]).count()
	while iters:
		newpr=data.iloc[:,0].groupby(data.iloc[:,1]).apply(lambda x:(pageranks/links)[x].sum()*damp+minpr)
		pageranks.update(newpr)
		iters-=1

	return pageranks

'''
data.columns=['wordid','from_urlid','to_urlid']
'''
#利用链接文本
def scoreLinkText(data,pageranks):
	scores=data.iloc[:,1].groupby(data.iloc[:,2]).apply(lambda x:pageranks[x].sum())
	return scores
