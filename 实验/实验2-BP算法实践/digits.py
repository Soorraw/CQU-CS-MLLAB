import numpy as np
from sklearn.datasets import load_digits
from template import error_BackPropagation as eBP
from template import Minmax_normalization as norm

digits=load_digits()
X=digits['data']#样本集
Y=digits['target']#标签集
grades=[[1,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,1]]#将k个无序属性值转化为k维向量
X=norm(X)

X_train=[]#训练集
Y_train=[]
X_test=[]#验证集
Y_test=[]
for i in range(X.shape[0]):#按7:3的比例将样本分为训练集和验证集
    if(i%10<7):
        X_train.append(X[i])
        Y_train.append(grades[Y[i]])
    else:
        X_test.append(X[i])
        Y_test.append(grades[Y[i]])


it=eBP(X=np.array(X_train),Y=np.array(Y_train),q=30,eta1=0.8,eta2=0.8,max_round=100)#初始化学习器,隐层神经元数目=30,η1=0.8,η2=0.8
it.BP(np.array(X_test),np.array(Y_test))#开始训练并记录在验证集上的验证结果
it.Show()#打印结果
it.Draw()#绘制图像