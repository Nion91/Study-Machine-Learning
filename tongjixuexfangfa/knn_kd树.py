# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame,Series

class knnKD:
    def selectDim(self,x):
        var=x.var(axis=0)
        sequence=var.argsort()[::-1]
        return sequence

    def selectMedian(self,values):
        length=len(values)
        if length%2==1:
            v=median(values)
        else:
            values.sort()
            v=values[length/2]
        return v

    #生成kd树
    def getTree(self,x,deep=0):
        if deep==0:
            self.dim_sequence=self.selectDim(x) 

        if x.shape[0]<=1:
            kdTree={'deep':deep,
                    'di':None,
                    'dv':None,
                    'leaf':True,
                    'node':x.flatten(),
                    'left':None,
                    'right':None}
            return kdTree
        
        split_index=self.dim_sequence[deep%len(self.dim_sequence)]
        dimvalues=Series(x[:,split_index])
        split_value=self.selectMedian(dimvalues.tolist())
        node=x[dimvalues[dimvalues==split_value].index[0]]

        kdTree={'deep':deep,
                'di':split_index,
                'dv':split_value,
                'leaf':False,
                'node':node,
                'left':self.getTree(x[x[:,split_index]<split_value],deep+1),
                'right':self.getTree(x[x[:,split_index]>split_value],deep+1)}

        if deep==0:
            self.kdTree=kdTree
        else:
            return kdTree

    #生成索引路径
    def searchPath(self,x,tree=None):
        nodes=list()
        if tree is None:
            tree=self.kdTree

        while True:
            nodes.append(tree['node'])
            if tree['leaf']:
                break
            elif x[tree['di']]<=tree['dv']:
                tree=tree['left']
            else:
                tree=tree['right']
        return nodes

    def getChildTree(self,parents,tree=None):
        childtree=self.kdTree.copy() if tree is None else tree.copy()
        for parent in parents[1:]:
            if (parent==childtree['left']['node']).all():
                childtree=childtree['left']
            else:
                childtree=childtree['right']
        return childtree

    def distance(self,x,y):
        dt=np.linalg.norm(x-y)
        return dt

    def getNeighbour(self,x,tree=None):
        nodes=self.searchPath(x,tree)

        #初始化近邻
        neighbour=current_node=nodes.pop()
        if neighbour.shape[0]==0:
            if len(nodes)==0:
                return neighbour
            neighbour=nodes[-1]
        dist=self.distance(x,neighbour)

        #更新近邻
        while len(nodes)>0:
            ct=self.getChildTree(nodes,tree)
            #在父节点的另一个子节点中进行搜索
            if abs(x[ct['di']]-ct['dv'])<dist:
                another_ct=ct['left'] if current_node.tolist()==ct['right']['node'].tolist() else ct['right']
                temp_nb=self.getNeighbour(x,another_ct)
                if temp_nb.shape[0]>0:
                    temp_dist=self.distance(x,temp_nb)
                    if dist>temp_dist:
                        neighbour=temp_nb
                        dist=temp_dist
            #判断父节点
            current_node=nodes.pop()
            parent_dist=self.distance(x,current_node)
            if dist>parent_dist:
                neighbour=current_node
                dist=parent_dist

        return neighbour


if __name__=='__main__':
    samples=np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
    kd=knnKD()
    kd.getTree(samples)
    for point in [(2.1,3.1),(2,3.9),(2,4.5),(8,8)]:
        print kd.getNeighbour(point)
