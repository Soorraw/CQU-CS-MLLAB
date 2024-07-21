import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import queue as que

def Holdout(dataset,hold=0.7):#根据留出法返回数据集dataset上的训练集D和验证集T
    dataset.drop_duplicates(inplace=True)#删除dataset上的重复数据
    D=dataset.sample(frac=hold)#提取训练集
    T=pd.concat([dataset,D]).drop_duplicates(keep=False)#以补集的方式提取验证集(A-B=pd.concat([A,B,B]).drop_duplicates(keep=False))
    return D,T

class Node:
    def __init__(self,D=None,a=None,t=None,label=None,value=None,depth=None,left=None,right=None):
        self.D=D#当前结点上的训练集
        self.a=a#划分属性
        self.t=t#划分点
        self.label=label#分类树叶子结点上的标签值
        self.value=value#回归树叶子结点上的输出值
        self.depth=depth#结点深度
        self.left=left#左结点
        self.right=right#右结点

    def isleaf(self):#检查该结点是否为叶子结点
        return (self.left==None) and (self.right==None)#没有左右子树则为叶子结点
    
    def beCleaf(self):#将当前结点转化为分类树型叶子结点
        label=self.D[self.D.columns[-1]]
        self.label=label.value_counts().index[0]#类别标记为当前结点训练集上样本数最多的类
        self.a=self.t=None#删除划分属性和划分属性值
        self.left=self.right=None#删除左右子树

    def beRleaf(self):#将当前结点转化为回归树型叶子节点
        value=self.D[self.D.columns[-1]]
        self.value=np.sum(value)/len(value)#输出值标记为当前结点训练集上的样本均值
        self.a=self.t=None#删除划分属性和划分属性值
        self.left=self.right=None#删除左右子树

    def setsub(self,a,t,subtrees):#将当前结点标记为内部结点并设置左右子树
        self.a=a#记录最优划分属性
        self.t=t#记录最优划分属性值
        self.label=self.value=None#清空标签值和输出值
        self.left=subtrees[0]#设置左子树
        self.right=subtrees[1]#设置右子树


class CART:
    def __init__(self,D,Regression=False,Preprune=True,delta=0,MinSet=5,MaxDepth=30,MaxNode=1e9):
        self.delta=delta#允许的最大误差(分类树为基尼值,回归树为方差和)
        self.regression=Regression#决策树树的类型,False为分类树,True为回归树
        self.preprune=Preprune#预剪枝,False为关,True为开
        self.MinSet=MinSet#预剪枝时的结点样本数阈值
        self.MaxDepth=MaxDepth#预剪枝时的决策树深度阈值
        self.MaxNode=MaxNode#决策树树的最大结点深度,避免内存超限
        self.root=Node(D,depth=0)#决策树的根节点
    
    def PrintTree(self):#打印决策树
        print("{",end="")
        self.PrintNode(self.root)
        print("}")

    def PrintNode(self,it):#打印决策树各结点
        if it.isleaf():#如果是叶子结点，则打印标签或输出值
            if self.regression==0:
                print("(label=%d)"%(it.label),end="")
            else: print("(value=%.2f)"%(it.value),end="")
            return
        
        print("%s<%.2f"%(it.a,it.t),end="")#如果是内部结点,打印划分依据
        if it.left.isleaf()==0:#左子树不是叶子节点则打印{}
            print('{',end="")
        self.PrintNode(it.left)#递归打印左子树
        if it.left.isleaf()==0:
            print('}',end="")

        print("%s>%.2f"%(it.a,it.t),end="")
        if it.right.isleaf()==0:#右子树不是叶子节点则打印{}
            print('{',end="")
        self.PrintNode(it.right)#递归打印右子树
        if it.right.isleaf()==0:
            print('}',end="")

    def Gini(self,D):#训练集D的基尼值
        pk=D[D.columns[-1]].value_counts(1)
        return 1-np.sum(np.square(pk))

    def Gini_index(self,D,a,t):#训练集D根据划分属性a和属性值t划分后的基尼指数
        D1=D[D[a]<t]
        D2=D[D[a]>t]
        return self.Gini(D1)/D1.shape[0]+self.Gini(D2)/D2.shape[0]

    def Variance(self,D,a,t):#训练集D根据划分属性a和属性值t划分后的方差和
        D1=D[D[a]<t]
        D2=D[D[a]>t]
        return np.var(D1[D1.columns[-1]])+np.var(D2[D2.columns[-1]])

    def Candidates(self,a):#连续性属性的候选划分点
        a=np.sort(a.value_counts().index)
        return (a[:-1]+a[1:])/2
    
    def BestCat(self,D,A):#分类(C)树的最优划分属性a和属性值t
        mgi=1
        for a in A:
            Ta=self.Candidates(D[a])
            for t in Ta:
                gi=self.Gini_index(D,a,t)#计算每个划分属性a和属性值t下的基尼指数
                if gi<mgi:
                    mgi=gi
                    bestfeature=a
                    bestvalue=t
        return bestfeature,bestvalue#返回使基尼指数最小化的a,t取值
    
    def BestRat(self,D,A):#回归(R)树的最优划分属性a和属性值t
        mvar=float(0x3f3f3f3f)
        for a in A:
            Ta=self.Candidates(D[a])
            for t in Ta:
                var=self.Variance(D,a,t)#计算每个划分属性a和属性值t下的方差和
                if var<mvar:
                    mvar=var
                    bestfeature=a
                    bestvalue=t
        return bestfeature,bestvalue#返回使方差和最小化的a,t取值
    
    def TreeGenerate(self):#BFS生成决策树
        nodes=1#初始化结点数为1(根节点)
        queue=que.Queue()
        queue.put(self.root)#创建队列并将根节点加入队列中

        while(not queue.empty()):#队列非空
            
            it=queue.get()#取出队头结点
            D=it.D#队头结点上的训练集
            A=D.columns[:-1]#训练集属性
            data=D[A]#样本属性值
            lv=D[D.columns[-1]]#样本标签或样本值
            
            if self.preprune:#如果启用预剪枝
                if len(data)<self.MinSet or it.depth>self.MaxDepth:#如果D中样本足够少或当前结点深度足够大,则将当前结点标记为叶结点
                    if self.regression==0:
                        it.label=lv.value_counts().index[0]#对分类树型结点,类别标记为D中样本数最多的类
                    else: it.value=np.sum(lv)/len(lv)#对回归树型结点,输出值为D中样本输出值的均值
                    continue
                if self.regression==0 and self.Gini(D)<self.delta:#如果结点为分类树型且基尼值小于阈值,则将当前结点标记为叶结点
                    it.label=lv.value_counts().index[0]#将当前结点标记为叶结点,类别标记为D中样本数最多的类
                    continue
                if self.regression==1 and np.var(D[D.columns[-1]])<self.delta:#如果结点为分类树型且方差和小于阈值,则将当前结点标记为叶结点
                    it.value=np.sum(lv)/len(lv)#将当前结点标记为叶结点,输出值为D中样本输出值的均值
                    continue

            if nodes>self.MaxNode or len(A)==0 or len(data.value_counts())==1:#如果结点数超过上限或属性集A为空或D中样本在A上取值相同
                if self.regression==0:#则依据结点类型标记为叶子结点
                    it.label=lv.value_counts().index[0]
                else: it.value=np.sum(lv)/len(lv)
                continue
            
            if len(lv.value_counts())==1:#如果D中样本全属于同一类别C,或D中样本输出值均为C
                if self.regression==0:
                    it.label=lv.iloc[0]#则将当前结点标记为C类叶节点
                else: it.value=lv.iloc[0]#或将当前结点输出值设置为C
                continue

            a,t=self.BestCat(D,A) if self.regression==0 else self.BestRat(D,A)#在A中选择最优划分属性a和最优划分属性值t

            D1=D[D[a]<t].copy()#划分左子树的样本子集D1
            D2=D[D[a]>t].copy()#划分右子树的样本子集D2
            if len(D1[a].value_counts())==1:#如果样本子集在属性值a上取值相同
                D1.drop(columns=a,inplace=True)#则删除样本子集中的该属性
            if len(D2[a].value_counts())==1:
                D2.drop(columns=a,inplace=True)
                
            it.setsub(a,t,[Node(D1,depth=it.depth+1),Node(D2,depth=it.depth+1)])#将当前结点标记为内部结点并设置左右子树
            queue.put(it.left)#左子结点入队列
            queue.put(it.right)#右子结点入队列
            nodes+=2   

    def Predict(self,i,it=None):#递归预测样例i的标签或输出值
        if it==None:#从根结点开始递归预测
            it=self.root
        if it.isleaf():#如果当前结点是叶子结点
            return it.label if self.regression==0 else it.value#则返回标签值或输出值
        return self.Predict(i,it.left) if i[it.a]<=it.t else self.Predict(i,it.right)#否则跟据划分属性和划分属性值继续递归预测
    
    def Accuracy(self,T):#返回分类树在验证集T上的分类准确率
        if self.regression==1:#回归树类型的对象不能调用本函数
            return 'Error:Regression Tree!'
        
        cnt=0#分类正确的样本数
        rows=T.shape[0]#样本数
        for i in range(rows):
            sample=T.iloc[i]#对每一行的样本
            if self.Predict(sample)==sample[-1]:#如果预测标签值与真实标签值相等
                cnt=cnt+1#分类正确的样本数增加1
        return cnt/rows#返回分类准确率
    
    def RMSE(self,T):#返回回归树在验证集T上的标准误差
        if self.regression==0:#分类树类型的对象不能调用本函数
            return 'Error:Classification Tree!'
        
        value=T[T.columns[-1]]#样本的真实输出
        predict=[]
        rows=T.shape[0]#样本数
        for i in range(rows):
            predict.append(self.Predict(T.iloc[i]))#获得每个样本的预测输出
        return np.sqrt(np.sum(np.square(np.array(predict)-value))/len(value))#返回预测输出和真实输出的标准误差

    def Postpruning(self,T,it=None):#后剪枝
        if(it==None):#从根结点开始递归剪枝
            it=self.root

        if(it.isleaf()):#如果当前结点是叶子结点,则直接返回
            return
        self.Postpruning(T,it.left)#对左子树进行剪枝
        self.Postpruning(T,it.right)#对右子树进行剪枝
        
        a=it.a#预处理,记录当前结点的划分属性和划分属性值
        t=it.t
        subtrees=[it.left,it.right]#记录当前结点的左右子树
        
        if self.regression==0:#对于分类树
            acc=self.Accuracy(T)#计算其不剪枝时在验证集T上的分类准确率
            it.beCleaf()#将当前结点替换为叶节点
            if self.Accuracy(T)>=acc:#如果替换后的分类树的分类准确率大于等于不剪枝时的分类准确率
                return#则确认剪枝操作
        else:#对于回归树
            rmse=self.RMSE(T)#计算其不剪枝时在验证集T上的标准误差
            it.beRleaf()#将当前结点替换为叶节点
            if self.RMSE(T)<=rmse:#如果替换后的回归树的标准误差小于等于不剪枝时的标准误差
                return#则确认剪枝操作
            
        it.setsub(a,t,subtrees)#否则不剪枝,恢复当前结点