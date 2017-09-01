import time
import random
import math
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import random
import copy

#宿舍分配问题
class DormTask:
	def __init__(self,dorm,pref):
		self.dorm=dorm*2
		self.pref=pref
		self.domain=DataFrame({'lower':0,'upper':range(len(dorm)*2-1,-1,-1)})

	def decodeGene(self,gene):
		dorm=copy.copy(self.dorm)
		data=Series([dorm.pop(i) for i in gene],index=self.pref.index)
		return data

	def cost(self,gene):
		data=self.decodeGene(gene)
		cost=np.where(data==self.pref.iloc[:,0],0,np.where(data==self.pref.iloc[:,1],1,3))
		return cost.sum()



#宿舍室友分配问题
class RoommateTask:
    def __init__(self,dorm,pref):
        self.dorm=dorm*2
        self.pref=pref.sort_index()
        self.domain=DataFrame({'lower':0,'upper':range(len(dorm)*2-1,-1,-1)})

    def decodeGene(self,gene):
        dorm=copy.copy(self.dorm)
        data=DataFrame({'student':self.pref.index.tolist(),'room':[dorm.pop(i) for i in gene]})
        p1=data.groupby('room').apply(lambda x:x.student.iloc[0]).rename('student1')
        p2=data.groupby('room').apply(lambda x:x.student.iloc[1]).rename('student2')
        result=pd.concat([p1,p2],axis=1)
        return result

    def cost(self,gene):
        data=self.decodeGene(gene)
        d1=Series(data.student1.tolist(),index=data.student2)
        d2=Series(data.student2.tolist(),index=data.student1)
        data=d1.append(d2).sort_index()
        cost=np.where(data==self.pref.iloc[:,0],0,np.where(data==self.pref.iloc[:,1],1,3))
        return cost.sum()




#飞机行程问题
class FlightTask:
    def __init__(self,flight,travel):
        self.flight=flight
        self.dest=travel.dest[0]
        self.initTravel(travel)
        self.initDomain()

    def initTravel(self,travel):
        return_flight=travel.rename(columns={'origin':'dest','dest':'origin'})
        self.travel=pd.concat([travel,return_flight])[['people','origin','dest']].set_index([range(travel.shape[0]*2)])

    def initDomain(self):
        upper=self.flight.groupby(['origin','dest']).apply(lambda x:x.shape[0]-1).rename('upper')
        upper=pd.merge(self.travel,upper.reset_index(),on=['origin','dest']).upper
        self.domain=DataFrame({'lower':0,'upper':upper},index=upper.index)

    def transTime(self,times):
        mins=times.map(lambda x:time.strptime(x,'%H:%M')).map(lambda x:x.tm_hour*60+x.tm_min)
        return mins

    def decodeGene(self,gene):
        temp=self.travel.copy()
        temp['id']=gene

        flight=self.flight.groupby(['origin','dest'],group_keys=False).apply(lambda x:x.iloc[temp.loc[(temp.origin==x.origin.iloc[0])&(temp.dest==x.dest.iloc[0]),'id']])
        flight=pd.merge(self.travel,flight,on=['origin','dest'])
        p1=flight.loc[flight.dest==self.dest].set_index('people').rename(columns={'depart':'ob_depart','arrive':'ob_arrive','price':'ob_price'})
        p2=flight.loc[flight.origin==self.dest].set_index('people').rename(columns={'depart':'rt_depart','arrive':'rt_arrive','price':'rt_price'})
        result=pd.concat([p1[['origin','ob_depart','ob_arrive','ob_price']],p2[['rt_depart','rt_arrive','rt_price']]],axis=1)

        return result

    def cost(self,gene):
        data=self.decodeGene(gene)

        #机票价格
        c1=data.ob_price.sum()+data.rt_price.sum()

        #等候时间
        arrive_time=self.transTime(data.ob_arrive)
        depart_time=self.transTime(data.rt_depart)
        c2=(arrive_time.max()-arrive_time).sum()+(depart_time-depart_time.min()).sum()

        #额外包车
        c3=50 if arrive_time.max()>=depart_time.min() else 0


        total=c1+c2+c3
        return total



#随机生成gene
def randomGene(task):
    gene=task.domain.apply(lambda x:random.randint(x.lower,x.upper),axis=1)
    return gene


#随机搜索
def randomOptimize(task,n=1000):
	result=None
	res_cost=float('Inf')
	while n:
		temp=randomGene(task)
		temp_cost=task.cost(temp)
		if temp_cost<res_cost:
			result=temp
			res_cost=temp_cost
		n-=1
	return result

#爬山法
def hillclimbOptimize(task,step=1):
	result=randomGene(task)
	while True:
		current=result.copy()
		for i in range(len(result)):
			for s in [step,-step]:
				num=result[i]+s
				if num<task.domain.lower[i] or num>task.domain.upper[i]:
					continue
				temp=current.copy()
				temp[i]=num
				if task.cost(temp)<task.cost(result):
					result=temp
		if (result==current).all():
			break
	return result

#模拟退火算法
def annealingOptimize(task,T=10000.0,cool=0.95,step=1):
	result=randomGene(task)
	while T>1.0:
		while True:
			i=random.randint(0,len(result)-1)
			s=random.choice([step,-step])
			num=result[i]+s
			if num<task.domain.lower[i] or num>task.domain.upper[i]:
				continue
			else:
				break
		temp=result.copy()
		temp[i]=num
		delta_cost=task.cost(temp)-task.cost(result)
		if delta_cost<0 or random.random()<pow(math.e,-delta_cost/T):
			result=temp
		T=T*cool
	return result

#遗传算法
def geneticOptimize(task,size=100,eliteprob=0.2,mutprob=0.2,step=1,iters=100):
	population=[randomGene(task) for i in range(size)]
	population.sort(key=lambda x:task.cost(x))
	while iters:
        #选取优秀基因
		elite=population[:int(size*eliteprob)]
		descendant=[]
		while len(descendant)<size-len(elite):
            #基因突变
			if random.random()<mutprob:
				temp=geneMutate(random.choice(elite),task.domain,step)
			#基因组合
            else:
				temp=geneCross(random.choice(elite),random.choice(elite))
			descendant.append(temp)
		population=elite+descendant
		population.sort(key=lambda x:task.cost(x))
		iters-=1
	return population[0]

#基因突变
def geneMutate(gene,domain,step=1):
	while True:
		i=random.randint(0,len(gene)-1)
		s=random.choice([step,-step])
		num=gene[i]+s
		if num<domain.lower[i] or num>domain.upper[i]:
			continue
		else:
			break
	mutated=gene.copy()
	mutated[i]=num
	return mutated

#基因组合
def geneCross(gene1,gene2):
    i=random.randint(1,len(gene1)-1)
    new=gene1.copy()
    new[i:]=gene2[i:]
    return new


if __name__=='__main__':
	#studentrooms
	dorms=['Z','A','H','B','P']
	prefs=DataFrame({'first_choice':['B','Z','A','Z','A','H','P','B','B','H'],
					 'second_choice':['H','P','Z','P','B','P','A','H','H','A']},
					 index=['Toby','Steve','Andrea','Sarah','Dave','Jeff','Fred','Suzie','Laura','Neil'])
	dt=DormTask(dorms,prefs)

    for i in range(100):
        t1=geneticOptimize(dt)
        if dt.cost(t1)<=2: break

    #roommate
    dorms=['Z','A','H','B','P']
    prefs=DataFrame({'first_choice':['Suzie','Neil','Fred','Laura','Andrea','Fred', 'Sarah', 'Andrea', 'Sarah', 'Steve'],
                     'second_choice':['Laura', 'Fred', 'Sarah', 'Jeff', 'Steve','Laura', 'Toby', 'Dave', 'Sarah', 'Jeff']},
                     index=['Toby','Steve','Andrea','Sarah','Dave','Jeff','Fred','Suzie','Laura','Neil'])
    rt=RoommateTask(dorms,prefs)

    #flights
    travel=DataFrame({'people':['Seymour','Franny','Zooey','Walt','Buddy','Les'],
                      'origin':['BOS','DAL','CAK','MIA','ORD','OMA'],
                      'dest':'LGA'}).loc[:,['people','origin','dest']]
    flights=pd.read_table(r'C:\Users\Administrator\Desktop\draft\data\schedule.txt',sep=',',names=['origin','dest','depart','arrive','price'])
    ft=FlightTask(flights,travel)
