from math import sqrt
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import copy


#把字典转为dataframe
def dict_to_df(prefs):
	sets=set()
	for value in prefs.values():
		sets=sets|set(value.keys())

	df=DataFrame({'id':prefs.keys()})
	for i in sets:
		temp=DataFrame({i:[value.get(i) for value in prefs.values()]})
		df=pd.concat([df,temp],axis=1)

	return df.set_index('id')


#计算欧几里得距离
def sim_distance(x1,x2):
	intersection=x1.index.intersection(x2.index)
	if intersection.shape[0]==0:
		return 0
	else:
		distance=(x1[intersection]-x2[intersection]).pow(2).sum()
		return 1/(1+sqrt(distance))


#计算皮尔逊相关系数
def sim_pearson(x1,x2):
	intersection=x1.index.intersection(x2.index)
	if intersection.shape[0]==0:
		return 0
	else:
		sim=x1[intersection].corr(x2[intersection])
		return sim


#按相似度匹配
def topMatches(prefs,x,n=10,similarity=sim_pearson):
	target=prefs.loc[x]
	sims=prefs.loc[prefs.index!=x].apply(lambda x:similarity(x,target),axis=1)
	sims=sims[sims>=0]
	return sims.sort_values(ascending=False).iloc[0:min(n,len(sims))]


#user_cf推荐
def getRecommendations(prefs,x,n=10,similarity=sim_pearson):
	sim=topMatches(prefs,x,n=n,similarity=similarity)
	subsets=prefs.loc[sim.index,np.isnan(prefs.loc[x])]
	rmd=subsets.apply(lambda x:(x*sim).sum()/(~np.isnan(x)*sim).sum() if x.any() else 0)
	return rmd.sort_values(ascending=False)


#user-item转为item-user
def transformPrefs(prefs):
	return prefs.T


#取得物品相似度表
def calculateSimilarItems(prefs,n=10,similarity=sim_distance):
	prefs_item=transformPrefs(prefs)
	result=DataFrame()
	for i in list(prefs_item.index):
		temp=topMatches(prefs_item,i,n,similarity)
		temp.name=i
		result=pd.concat([result,temp],axis=1)
	return result


#item_cf推荐
def getRecommendedItems(prefs,itemMatch,user):
	score=prefs.loc[user].dropna()
	weights=itemMatch.loc[score.index,~itemMatch.columns.isin(score.index)]
	sim=weights.apply(lambda x:(x*score).sum()/x.sum() if x.any() else 0).sort_values(ascending=False)
	return sim



if __name__=='__main__':
	critics={'Lisa Rose':{'Lady in the Water':2.5,
						  'Snakes on a Plane':3.5,
						  'Just My Luck':3.0,
						  'Superman Returns':3.5,
						  'You, Me and Dupree':2.5,
						  'The Night Listener':3.0},
			 'Gene Seymour':{'Lady in the Water':3.0,
						  'Snakes on a Plane':3.5,
						  'Just My Luck':1.5,
						  'Superman Returns':5.0,
						  'You, Me and Dupree':3.5,
						  'The Night Listener':3.0},
			 'Michael Phillips':{'Lady in the Water':2.5,
						  'Snakes on a Plane':3.0,
						  'Superman Returns':3.5,
						  'The Night Listener':4.0},
			 'Claudia Puig':{'Snakes on a Plane':3.5,
						  'Just My Luck':3.0,
						  'Superman Returns':4.0,
						  'You, Me and Dupree':2.5,
						  'The Night Listener':4.5},
			 'Mic LaSalle':{'Lady in the Water':3.0,
						  'Snakes on a Plane':4.0,
						  'Just My Luck':2.0,
						  'Superman Returns':3.0,
						  'You, Me and Dupree':2.0,
						  'The Night Listener':3.0},
			 'Jack Matthews':{'Lady in the Water':3.0,
						  'Snakes on a Plane':4.0,
						  'Superman Returns':5.0,
						  'You, Me and Dupree':3.5,
						  'The Night Listener':3.0},
			 'Toby':{'Snakes on a Plane':4.5,
						  'Superman Returns':4.0,
						  'You, Me and Dupree':1.0}}

	prefs=dict_to_df(critics)
	getRecommendations(prefs,'Toby')

	itemsim=calculateSimilarItems(prefs)
	getRecommendedItems(prefs,itemsim,'Toby')
