import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from template import load_iris#鸢尾花数据集
from template import Logistic_Regression as LR#导入对率回归分类器模板

def mode(result,k):#基于计数排序求长度为k的list对象result中的众数
    counts=[0 for i in range(k)]
    for i in result:
         counts[i]=counts[i]+1
    times=0
    for i in range(k):
         if counts[i]>times:
              times=counts[i]
              ans=i
    return ans
iris=load_iris()
iris_data=iris['data']#样本集
iris_target=iris['target']#标签集
k=len(iris['target_names'])#多分类学习的类别数
X_train=[[] for i in range(k)]#训练集
X_test=[]#验证集
y_test=[]
for i in range(len(iris_data)):#按7:3的比例划分训练集和验证集
    if(i%10<7):
        X_train[iris_target[i]].append(iris_data[i])#X_train[i]中记录的是类标签为i的训练样本
    else:
        X_test.append(iris_data[i])
        y_test.append(iris_target[i])
n=int(k*(k-1)/2)#拆分后的二分类器数量
classify=[[] for i in range(n)]#chassify[i]中记录的是第i个0/1二分类器分类结果的真实标签
cnt=0
times1=200#循环次数
times2=10#每次循环所有二分类器进行梯度下降的次数
lrs=[]#记录所有的二分类器
print("基于标准正态分布生成的%d个分类器的初始权值β分别为"%(n))
for i in range(k):
    for j in range(i+1,k):
            classify[cnt].append(i)#被第cnt个二分类器分类为0的样本真实标签为i
            classify[cnt].append(j)#被第cnt个二分类器分类为1的样本真实标签为j
            cnt=cnt+1  
            X=X_train[i].copy()
            list.extend(X,X_train[j])#将真实标签为i的样本集合真实标签为j的样本集进行组合
            y=[0 for it in range(len(X_train[i]))]
            list.extend(y,[1 for it in range(len(X_train[j]))])#将这些样本分别打上0/1标签
            lr=LR(np.array(X),np.array(y))#利用这些样本集和标签集对第cnt个二分类学习器进行初始化
            lrs.append(lr)#记录该分类器
            print(lr.beta)

x=[]#记录迭代次数/横坐标
fx=[]#记录准确率/纵坐标
l=len(y_test)#验证集样本数
X_test=np.array(X_test)
X_test=np.insert(X_test,X_test.shape[1],1,axis=1)#对验证集所有样本的属性进行拓展
for i in range(times1):
    for j in range(n):
        lrs[j].Grad_descent(times2)#每次循环进行times2次梯度下降
    betas=[]    
    for j in range(n):
        betas.append(np.array(lrs[j].beta))#记录当前每个二分类器的权值向量β
    results=[[] for i in range(l)]
    cnt=0#记录分类正确的验证集样本个数
    for j in range(l):
        for t in range(len(betas)):
            if (np.dot(X_test[j],betas[t])>0):#判断当前每个0/1二分类器的分类结果
                idx=1
            else: idx=0
            results[j].append(classify[t][idx])#将每个分类器的分类结果映射为对应标签并记录
        result=mode(results[j],k)#计算投票结果
        if(result==y_test[j]):#投票结果与真实标签如果相同则分类正确
            cnt=cnt+1
    x.append(times2*(i+1))#记录迭代次数
    fx.append(cnt/l*100)#记录在多分类学习器在验证集上的分类准确率
print("迭代%d次后的权值β分别为"%(times1*times2))#打印最终每个二分类器的权值向量β
for i in range(n):
    print(betas[i])
print("在验证集上的分类准确率为%.2f%%"%(fx[-1]))#打印最终的分类准确率
plt.plot(x,fx)#数据可视化
plt.xlabel('Iterations(number of times)')#横坐标为迭代次数
plt.ylabel('Accuracy(%)')#纵坐标为在验证集上的分类准确率
plt.show()