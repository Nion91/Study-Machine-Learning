import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from math import sqrt

###层次化聚类
class Bicluster:
	def __init__(self,vec,distance=0,left=None,right=None,index=None):
		self.vec=vec
		self.distance=distance
		self.left=left
		self.right=right
		self.index=index


def distancePearson(data):
	sim=data.apply(lambda x:data.corrwith(data.loc[x.name],axis=1),axis=1)
	return 1-sim

def minDistance(distmatrix):
	min_dists=distmatrix.apply(lambda x:x[x.index!=x.name].min())
	x=min_dists.argmin()
	y=distmatrix[x][distmatrix[x].index!=x].argmin()
	min_dist=min_dists.min()
	return x,y,min_dist


def hcluster(data,cluster=None,distmatrix=None,groupid=None):
	'''
	data为数据集，用于计算距离矩阵
	cluster为未聚类的Bicluster
	distmatrix为距离矩阵
	'''

	if cluster is None:
		cluster=data.apply(lambda x:Bicluster(np.array(x),index=x.name),axis=1)
	elif len(cluster)==1:
		return cluster[0]

	if distmatrix is None:
		distmatrix=distancePearson(data)
	if groupid is None:
		groupid='g1'

	#找到当前距离最近的两个Bicluster
	left,right,min_dist=minDistance(distmatrix)
	left=cluster[left]
	right=cluster[right]

	to_delete=[left.index,right.index]
	newvec=(left.vec+right.vec)/2.0

	#聚类生成一个新的Bicluster
	newone=Bicluster(newvec,distance=min_dist,left=left,right=right,index=groupid)

	#更新data,cluster,dismatrix，以便递归
	newdata=pd.concat([data.loc[~data.index.isin(to_delete)],DataFrame({groupid:newvec},index=data.columns).T])
	newcluster=cluster[~cluster.index.isin(to_delete)].append(Series([newone],index=[groupid]))

	newdist=1-newdata.corrwith(newdata.loc[groupid],axis=1)
	newdistmatrix=distmatrix.loc[~distmatrix.index.isin(to_delete),~distmatrix.columns.isin(to_delete)]
	newdistmatrix=pd.concat([newdistmatrix,DataFrame({groupid:newdist[newdist.index!=groupid]})],axis=1)
	newdistmatrix=pd.concat([newdistmatrix,DataFrame({groupid:newdist}).T])

	newgroupid='g%s' % (int(groupid[1:])+1)

	#递归
	return hcluster(newdata,newcluster,newdistmatrix,newgroupid)

#打印树
def printCluster(cluster,indent=0):
	if cluster.left is None:
		print '  '*indent,'--',cluster.index
	else:
		printCluster(cluster.left,indent+1)
		printCluster(cluster.right,indent+1)

#可视化聚类数
from PIL import Image,ImageDraw

def getHeight(clust):
	if clust.left==None: 
		return 1
	else:
		return getHeight(clust.left)+getHeight(clust.right)

def getDepth(clust):
	if clust.left==None:
		return 0
	else:
		return max(getDepth(clust.left),getDepth(clust.right))+clust.distance

def drawDendrogram(clust,jpeg='cluster.jpg'):
	h=getHeight(clust)*20
	w=1200
	d=getDepth(clust)

	scaling=float(w-150)/d

	img=Image.new('RGB',(w,h),(255,255,255))
	draw=ImageDraw.Draw(img)
	draw.line((0,h/2,10,h/2),fill=(255,0,0))

	drawnode(draw,clust,10,(h/2),scaling)
	img.save(jpeg,'JPEG')

def drawNode(draw,clust,x,y,scaling):
	if clust.left is not None:
		h1=getHeight(clust.left)*20
		h2=getHeight(clust.right)*20
		top=y-(h1+h2)/2
		bottom=y+(h1+h2)/2

		l1=clust.distance*scaling
		draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))
		draw.line((x,top+h1/2,x+l1,top+h1/2),fill=(255,0,0))
		draw.line((x,bottom-h2/2,x+l1,bottom-h2/2),fill=(255,0,0))

		drawNode(draw,clust.left,x+l1,top+h1/2,scaling)
		drawNode(draw,clust.right,x1+l1,bottom-h2/2,scaling)
	else:
		draw.text((x+5,y-7),clust.id,(0,0,0))



###k-means聚类
def initKCenter(data,k):
	ranges=data.apply(lambda x:(x.min(),x.max()))
	center=DataFrame({'center':range(k)}).groupby('center').apply(lambda x:ranges.apply(lambda x:random.uniform(x[0],x[1])))
	return center

def kcluster(data,k):
	centers=initKCenter(data,k)
	oldcluster=Series(None,index=data.index)

	iter_count=1
	while True:
		newcluster=data.apply(lambda x:1-centers.corrwith(x,axis=1),axis=1).apply(lambda x:x.argmin(),axis=1)
		if (oldcluster==newcluster).all():
			break
		oldcluster=newcluster
		centers=data.groupby(newcluster).apply(lambda x:x.mean(axis=0))
		iter_count+=1
	print iter_count

	return newcluster


###其他
#tamimot距离
def tamimoto(x,y):
	set_x=set(x[x==1].index.tolist())
	set_y=set(y[y==1].index.tolist())
	sim=len(set_x.intersection(set_y))/len(set_x.union(set_y))
	return 1-sim

def tamimoto_on_df(data,y):
	dist=data.apply(lambda x:tamimoto(x,y),axis=1)
	return dist

def distanceTamimoto(data):
	dist=data.apply(lambda x:tamimoto_on_df(data,x),axis=1)
	return dist

#欧几里得距离
def distanceEuclidean(data):
	dist=data.apply(lambda x:data.apply(lambda y:sqrt((x-y).pow(2).sum()),axis=1),axis=1)
	return dist



#多维缩放
def multiDimensionScaling(data,rate=0.01,dist=distancePearson):
	n=data.shape[0]
	loc=DataFrame(np.random.random(n*2).reshape((n,2)),index=data.index,columns=['x','y']) #随机生成坐标
	lasterror=None

	real_dist=dist(data) #真实的距离矩阵
	while True:
		fake_dist=distanceEuclidean(loc) #根据坐标计算出的距离矩阵
		err=(fake_dist-real_dist)/real_dist #移动比例因子矩阵

		totalerror=err.abs().sum().sum() #总误差
		if lasterror and lasterror<totalerror:
			break
		lasterror=totalerror

		loc_x_dist=loc.x.apply(lambda x:x-loc.x) #x坐标距离矩阵
		loc_y_dist=loc.y.apply(lambda y:y-loc.y) #y坐标距离矩阵

		grad_x=(loc_x_dist/fake_dist*err).sum(axis=1) 
		grad_y=(loc_y_dist/fake_dist*err).sum(axis=1)
		grad=DataFrame({'x':grad_x,'y':grad_y}) #坐标移动比例

		loc=loc-grad*rate #更新坐标，rate为移动速度

	return loc


def scatterText(data):
	plt.subplot(111)
	for i in range(data.shape[0]):
		plt.text(data.x.iloc[i],data.y.iloc[i],data.index.tolist()[i])
	plt.axis([-0.5,1.5,-0.5,1.5])
	plt.show()



if __name__=='__main__':
	blog=pd.read_table(r'C:\Users\Administrator\Desktop\draft\data\blogdata.txt').set_index('Blog')
	zebo=pd.read_table(r'C:\Users\Administrator\Desktop\draft\data\zebo.txt').set_index('Item')
