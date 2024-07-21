#读取本地数据
import csv
import numpy as np

#读取西瓜数据集中的数据并进行预处理
def loadDataset(filename):
    dataset=[]
    labelset=[]
    with open(filename,'r') as csvfile:
        csv_reader=csv.reader(csvfile)
        header=next(csv_reader)
        for row in csv_reader:
            if row[3] == '是':
                labelset.append(1)
            elif row[3] == '否':
                labelset.append(0)
            row[3]=1
            dataset.append(row)
    data=[[float(x) for x in row]for row in dataset]
    return dataset,labelset

#定义sigmoid函数
def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def test(dataset,labelset,w):
    data=np.mat(dataset).astype(float)

    y=sigmoid(np.dot(data,w))
    b,c=np.shape(y)#功能是查看矩阵或者数组的维数。
    rightcount=0

    for i in range(b):
        flag=-1
        if y[i,0]>0.5:
            flag=1
        elif y[i,0]<0.5:
            flag=0
        if labelset[i] == flag:
            rightcount+=1

    rightrate=rightcount/len(dataset)
    return rightrate

#迭代求w
def training(dataset,labelset):
    # np.dot(a,b) a和b矩阵点乘
    # np.transpose()  转置
    # np.ones((m,n))  创建一个m行n列的多维数组
    data=np.mat(dataset).astype(float)
    label=np.mat(labelset).transpose()
    w = np.ones((len(dataset[0]),1))

    #步长
    n=0.0001

    # 每次迭代计算一次正确率（在测试集上的正确率）
    # 达到0.90的正确率，停止迭代
    rightrate=0.0
    while rightrate<0.90:
        c=sigmoid(np.dot(data,w))
        b=c-label
        change = np.dot(np.transpose(data),b)
        w=w-change*n
        #预测，更新准确率
        rightrate = test(dataset,labelset,w)

    return w


dataset=[]
labelset=[]
filename = '西瓜数据集3.0a.csv'
dataset,labelset=loadDataset(filename)
w=training(dataset,labelset)
print("w和b分别为：\n",w)
#print("正确率：%f"%(test(dataset,labelset,w)*100)+"%")