import pandas as pd
from Template import CART
from Template import Holdout


linnerud=pd.read_csv("linnerud.txt")
D1,T1=Holdout(linnerud)#依据留出法将数据集划分为训练集D和验证集T

RT1=CART(D1,Regression=True,Preprune=False)#初始化决策树,类型为回归型,不启用预剪枝
RT1.TreeGenerate()#生成决策树
RT1.PrintTree()#打印决策树
print("体能训练数据集:未进行后剪枝的回归树的标准误差为%.2f"%(RT1.RMSE(T1)))
RT1.Postpruning(T1)#后剪枝
RT1.PrintTree()#打印决策树
print("体能训练数据集:剪枝后的回归树的标准误差为%.2f"%(RT1.RMSE(T1)))

# diabetes=pd.read_csv("diabetes.txt")
# D2,T2=Holdout(diabetes)

# RT2=CART(D2,Regression=True,MinSet=4,MaxDepth=20)#初始化决策树,类型为回归型,最小样本集阈值为4,最大深度为20
# RT2.TreeGenerate()
# RT2.PrintTree()
# print("糖尿病数据集:未进行后剪枝的回归树的标准误差为%.2f"%(RT2.RMSE(T2)))
# RT2.Postpruning(T2)
# RT2.PrintTree()
# print("糖尿病数据集:剪枝后的回归树的标准误差为%.2f"%(RT2.RMSE(T2)))