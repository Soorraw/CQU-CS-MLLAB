import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from template import Logistic_Regression as LR#导入对率回归分类器模板

file=pd.read_csv('exps\watermelon.csv',encoding='UTF-8')#读入西瓜数据集
file.drop(labels=["编号"], axis=1, inplace=True)#删除"编号"列
file["色泽"].replace(to_replace=["浅白","青绿","乌黑"], value=[0.0, 0.5, 1.0], inplace=True)#将离散属性转化为连续值
file["根蒂"].replace(to_replace=["蜷缩","稍蜷","硬挺"], value=[0.0, 0.5, 1.0], inplace=True)
file["敲声"].replace(to_replace=["清脆","沉闷","浊响"], value=[0.0, 0.5, 1.0], inplace=True)
file["纹理"].replace(to_replace=["清晰","稍糊","模糊"], value=[0.0, 0.5, 1.0], inplace=True)
file["脐部"].replace(to_replace=["平坦","稍凹","凹陷"], value=[0.0, 0.5, 1.0], inplace=True)
file["触感"].replace(to_replace=["硬滑","软粘"], value=[0.0, 1.0], inplace=True)
file["好瓜"].replace(to_replace=["否","是"], value=[0, 1], inplace=True)#给定标签
X=file.values[:,:-1]#样本集
y=file.values[:,-1]#标签集

X_train=[]#训练集
y_train=[]
X_test=[]#验证集
y_test=[]
for i in range(X.shape[0]):#按7:3的比例将样本分为训练集和验证集
    if(i%10<7):
        X_train.append(X[i])
        y_train.append(y[i])
    else:
        X_test.append(X[i])
        y_test.append(y[i])
times1=500#循环次数
times2=10#每次循环进行的梯度下降次数
x=[]#记录迭代次数/横坐标
fx=[]#记录准确率/纵坐标
cnt=0#迭代次数
lr=LR(np.array(X_train),np.array(y_train))#初始化对率回归分类器
X_test=np.array(X_test)
X_test=np.insert(X_test,X_test.shape[1],1,axis=1)#对验证集所有样本的属性进行拓展
for i in range(times1):
    lr.Grad_descent(times2) #每次循环进行times2次梯度下降
    results1=np.dot(lr.X,lr.beta)#训练集上的分类结果
    results2=np.dot(X_test,lr.beta)#验证集上的分类结果
    cnt1=0#训练集上正确分类次数
    cnt2=0#验证集上正确分类次数
    k1=len(y_train)
    k2=len(y_test)
    for j in range(k1):#计算训练集上正确分类次数
        if (results1[j]>0 and y_train[j]==1) or (results1[j]<0 and y_train[j]==0):
            cnt1=cnt1+1   
    for j in range(k2):#计算验证集上正确分类次数
        if (results2[j]>0 and y_test[j]==1) or (results2[j]<0 and y_test[j]==0):
            cnt2=cnt2+1
    cnt=cnt+times2
    if(cnt%(times1*times2/10)==0):#每times1*times2/10次迭代记录并打印一次分类结果   
        print("第%d次迭代,在训练集上分类准确率为%.2f%%,在验证集上分类准确率为%.2f%%"%((i+1)*times2,cnt1/k1*100,cnt2/k2*100))
        x.append(cnt)
        fx.append(cnt2/k2*100)
plt.plot(x,fx)#数据可视化
plt.xlabel('Iterations(number of times)')#横坐标为迭代次数
plt.ylabel('Accuracy(%)')#纵坐标为在验证集上的分类准确率
plt.show()