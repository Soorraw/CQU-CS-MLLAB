import pandas as pd
from Template import CART
from Template import Holdout


iris=pd.read_csv("iris.txt")
D1,T1=Holdout(iris)#依据留出法将数据集划分为训练集D和验证集T

CT1=CART(D1,Preprune=False)#初始化决策树,默认为分类树,不启用预剪枝
CT1.TreeGenerate()#生成决策树
CT1.PrintTree()#打印决策树
print("鸢尾花数据集:未进行后剪枝的分类树的分类准确率为%.2f%%"%(CT1.Accuracy(T1)*100))
CT1.Postpruning(T1)#后剪枝
CT1.PrintTree()#打印决策树
print("鸢尾花数据集:剪枝后的分类树的分类准确率为%.2f%%"%(CT1.Accuracy(T1)*100))

wine=pd.read_csv("wine.txt")
D2,T2=Holdout(wine)

CT2=CART(D2,delta=0.3,MinSet=4)#初始化决策树,默认启用预剪枝,最大可接受基尼值为0.3,最小样本集阈值为4
CT2.TreeGenerate()
CT2.PrintTree()
print("红酒数据集:未进行后剪枝的分类树的分类准确率为%.2f%%"%(CT2.Accuracy(T2)*100))
CT2.Postpruning(T2)
CT2.PrintTree()
print("红酒数据集:剪枝后的分类树的分类准确率为%.2f%%"%(CT2.Accuracy(T2)*100))