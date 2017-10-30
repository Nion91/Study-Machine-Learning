import numpy as np


def SVD(x,k=0.9):
    eigenvalue,eigenvector=np.linalg.eig(x.T.dot(x))
    index=eigenvalue.argsort()[::-1]
    Sigma=np.sqrt(eigenvalue[index])
    rates=(Sigma**2).cumsum()/(Sigma**2).sum()
    for i in range(rates.shape[0]):
        if k<=rates[i]:
            n=i
            break
    index=index[:(i+1)]
    Sigma=Sigma[:(i+1)]
    V=eigenvector[:,index]
    U=np.apply_along_axis(lambda i:x.dot(i),0,V)/Sigma

    return U,Sigma,V.T


#协同过滤
#相似度
def simEuclidean(x,y):
    return 1.0/(1+np.linalg.norm(x-y))

def simPearson(x,y):
    return 0.5+0.5*np.corrcoef(x,y,rowvar=0)[0,1]

def simCosine(x,y):
    return 0.5+0.5*x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))

#item相似度矩阵
def getItemMat(data,simFunc=simCosine):
    mat=np.apply_along_axis(lambda i:np.apply_along_axis(lambda j:simFunc(i,j),0,data),0,data)
    return mat

#矩阵SVD化
def svdData(data):
    U,Sigma,VT=SVD(data,0.9)
    result=data.T.dot(U).dot(np.linalg.inv(np.diag(Sigma)))
    return result

#推荐
def recommend(data,user,itemMat,n):
    itemScores=[]
    unratedItem=np.nonzero(data[user]==0)[0]
    ratedItem=np.nonzero(data[user])[0]
    for item in unratedItem:
        sim=itemMat[item,ratedItem]
        totalrate=data[user,ratedItem].dot(sim)
        totalsim=sim.sum()
        score=totalrate/totalsim if totalsim!=0 else 0
        itemScores.append((item,score))
    itemScores.sort(key=lambda x:x[1],reverse=True)
    if len(itemScores)>n:
        itemScores=itemScores[:n]
    return itemScores


#读取img数据
def read_img(filepath):
    result=[]
    for line in open(filepath).readlines():
        row=[]
        for i in range(32):
            row.append(int(line[i]))
        result.append(row)
    return result


#打印矩阵
def printMat(mat,threshold):
    for i in range(32):
        for j in range(32):
            if mat[i,j]>threshold:
                print 1,
            else:
                print 0,
        print ''



if __name__=='__main__':
    x=np.array([[1, 1, 0, 2, 2],
                [0, 0, 0, 3, 3],
                [0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0],
                [2, 2, 2, 0, 0],
                [5, 5, 5, 0, 0],
                [1, 1, 1, 0, 0]])

    SVD(x)

    #推荐
    x=np.array([[4, 4, 0, 2, 2],
                [4, 0, 0, 3, 3],
                [4, 0, 0, 1, 1],
                [1, 1, 1, 2, 0],
                [2, 2, 2, 0, 0],
                [1, 1, 1, 0, 0],
                [5, 5, 5, 0, 0]])

    itemMat=getItemMat(x,simPearson)
    recommend(x,2,itemMat,3)

    x=np.array([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])
    svdMat=getItemMat(svdData(x).T,simCosine)
    recommend(x,2,svdMat,3)

    #压缩图形
    filepath='D:\\Documents\\Downloads\\study\\machinelearninginaction\\Ch14\\'
    data=read_img(filepath+'0_5.txt')
    img=np.array(data)

    U,Sigma,VT=SVD(img,0.8)
    new_img=U.dot(np.diag(Sigma)).dot(VT)
    printMat(new_img,0.8)
