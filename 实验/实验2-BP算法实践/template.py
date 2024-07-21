import numpy as np
import matplotlib.pyplot as plt

def Minmax_normalization(X):#最大最小归一化
    for feature in range(X.shape[1]):
        max_value=max(X[:,feature])#计算样本集中每种属性的最大最小值
        min_value=min(X[:,feature])
        if max_value!=min_value:
            X[:,feature]=(X[:,feature]-min_value)/(max_value-min_value)#归一化
        else: X[:,feature]=0.5
    return X

class error_BackPropagation:
    def __init__(self,X,Y,q,eta1=0.1,eta2=0.1,delta=0.01,max_round=100):
        self.X=X#训练样本集
        self.Y=Y#训练标签集
        self.row=X.shape[0]#样本数
        self.d=X.shape[1]#输入层神经元数量等于样本属性数
        self.l=Y.shape[1]#输出层神经元数量等于标签属性数
        self.q=q#隐层神经元数量
        self.w=np.random.randn(self.q,self.l)#基于标准正态分布初始化连接权和阈值
        self.v=np.random.randn(self.d,self.q)
        self.theta=np.random.randn(self.l)
        self.gamma=np.random.randn(self.q)
        self.eta1=eta1#学习率1
        self.eta2=eta2#学习率2
        self.delta=delta#可接受的最大累积误差
        self.max_round=max_round#最大迭代轮数
        self.estimator=np.zeros([Y.shape[0],Y.shape[1]])#神经网络对于各个样本的输出
        self.round=[]#当前迭代次数
        self.accuracy=[]#当前分类准确率

    def Alpha(self,x):#隐层神经元的输入
        return np.matmul(x,self.v)
    
    def Beta(self,b):#输出神经元的输入
        return np.matmul(b,self.w)

    def Sigmoid(self,x):#激活函数
        return 1/(1+np.exp(-x))
    
    def g(self,y,estimator):#用于更新的参数g
        return estimator*(1-estimator)*(y-estimator)
    
    def e(self,b,g):#用于更新的参数e
        return b*(1-b)*np.matmul(self.w,g)
    
    def E(self,estimator,Y):#累积误差
        return np.sum(np.square(estimator-Y))/(2*self.row)
    
    def update(self,b,g,x,e):#对参数进行更新
        self.w+=self.eta1*np.outer(b,g)
        self.v+=self.eta2*np.outer(x,e)
        self.theta-=self.eta1*g
        self.gamma-=self.eta2*e

    def BP(self,X_test,Y_test):
        round=0
        isend=0      
        k=X_test.shape[0]#验证集样本数
        while round<self.max_round and not isend:#到达最大迭代轮数或满足停止条件则停止
            round+=1
            for it in range(self.row):
                x=self.X[it]#对于每一个训练集中的样本
                y=self.Y[it]
                alpha=self.Alpha(x)#计算各隐层神经元的输入
                b=self.Sigmoid(alpha-self.gamma)#计算各隐层神经元的输出
                beta=self.Beta(b)#计算各输出神经元的输入
                self.estimator[it]=self.Sigmoid(beta-self.theta)#计算各输出神经元的输出

                g=self.g(y,self.estimator[it])#计算用于更新的参数g
                e=self.e(b,g)#计算用于更新的参数e

                self.update(b,g,x,e)#更新连接权和阈值
                                
                isend=self.E(self.estimator,self.Y)<self.delta#终止条件1:当前累积误差小于可接受的最大累积误差
                if isend:
                    break   
                
            cnt=0
            for it in range(k):
                x=X_test[it]#对于每一个验证集中的样本,重复上述步骤
                y=Y_test[it]
                alpha=self.Alpha(x)
                b=self.Sigmoid(alpha-self.gamma)
                beta=self.Beta(b)
                estimator=self.Sigmoid(beta-self.theta)
                flag=1
                for j in range(self.l):#如果该样本在神经网络中的输出符合实际
                    if((estimator[j]>0.5)!=y[j]):
                        flag=0
                cnt+=flag
            accuracy=cnt/k*100#计算当前的分类准确率
            self.round.append(round)#记录当前迭代轮数
            self.accuracy.append(accuracy)#记录当前分类准确率
            if accuracy<90:#终止条件2:当前分类准确率大于90%
                isend=0
    
    def Draw(self):#绘制迭代轮数-分类准确率图像
        plt.plot(self.round,self.accuracy)
        plt.xlabel('Rounds(times)')
        plt.ylabel('Accuracy(%)')
        plt.show()
    
    def Show(self):#打印最终的迭代次数,累积误差和分类准确率
        print("标准BP算法经过%d轮迭代,累积误差为%.4f,分类准确率为%.2f%%"%(self.round[-1],self.E(self.estimator,self.Y),self.accuracy[-1]))