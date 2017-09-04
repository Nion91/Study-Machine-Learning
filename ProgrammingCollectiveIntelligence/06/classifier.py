import re
import copy
import math

class Classifier:
    def __init__(self):
        self.wc={}
        self.dc={}
        self.dwc={}
        self.yprob={}

    def train(self,x,y):
        #更新wordscount
        words=self.getWords(x)
        for k in words:
            if self.wc.get(k,0):
                self.wc[k][y]=self.wc[k].get(y,0)+1
            else:
                self.wc[k]={y:1}
                #更新每一类的wordcount
                self.dwc[y]=self.dwc.get(y,0)+1
        #更新documentcount
        self.dc[y]=self.dc.get(y,0)+1

    def getWords(self,text):
        pattern=re.compile('\\W*')
        words=[s.lower() for s in pattern.split(text) if len(s)>2]
        return dict([(word,1) for word in words])

    def initProb(self):
        total=sum([i for i in self.dc.values()])*1.0
        self.condprob=copy.deepcopy(self.wc)
        for k,v in self.dc.items():
            #更新类的概率
            self.yprob[k]=v/total
            for w in self.wc:
                #更新词的条件概率
                #self.condprob[w][k]=1.0*self.wc[w][k]/self.dc[k] if self.wc[w].get(k,0) else 1.0/total
                self.condprob[w][k]=1.0*(self.wc[w].get(k,0)+1)/(self.dc[k]+1)

    
#朴素贝叶斯
class BayesCF(Classifier):
    def prob(self,x):
        if len(self.yprob)==0: self.initProb()

        words=self.getWords(x)
        probs={}
        for i in self.dc:
            probs[i]=1
            for j in words:
                probs[i]=probs[i]*self.condprob[j][i]*self.yprob[i]
        return probs

    def predict(self,x,threshold=None):
        probs=[(k,v) for k,v in self.prob(x).items()]
        probs.sort(key=lambda x:x[1],reverse=True)
        if threshold:
            if probs[0][1]/probs[1][1]<threshold:
                return 'unknow'
        return probs[0][0]


#Fisher方法
class FisherCF(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.tcondprob={}
        self.limit={}

    def setLimit(self,y,num):
        self.limit[y]=num

    def initProb(self):
        Classifier.initProb(self)
        for i in self.condprob:
            #更新词的条件概率之和
            self.tcondprob[i]=sum([v for v in self.condprob[i].values()])

    def cprob(self,x,y):
        if len(self.yprob)==0: self.initProb()

        prob=self.condprob[x][y]/self.tcondprob[x]
        return prob

    def fscore(self,x,y):
        score=-2*math.log(self.cprob(x,y))
        return score

    def invchi2(self,chi,df):
        m=chi/2.0
        total=term=math.exp(-m)
        for i in range(1,df//2):
            term*=m/i
            total+=term
        return min(total,1)

    def fisherprob(self,x,y):
        words=self.getWords(x)
        score=sum([self.fscore(w,y) for w in words])
        return self.invchi2(score,len(words)*2)

    def prob(self,x):
        probs={}
        for i in self.dc:
            probs[i]=self.fisherprob(x,i)
        return probs

    def predict(self,x):
        probs=[(k,v) for k,v in self.prob(x).items()]
        probs.sort(key=lambda x:x[1],reverse=True)
        result=probs[0][0] if probs[0][1]>=self.limit.get(probs[0][0],0) else 'unknow'
        return result





if __name__='__main__':
    cf=FisherCF()

    cf.train('Nobody owns the water.','good')
    cf.train('the quick rabbit jumps fences','good')
    cf.train('buy pharmaceuticals now','bad')
    cf.train('make quick money at the online casino','bad')
    cf.train('the quick brown fox jumps','good')
