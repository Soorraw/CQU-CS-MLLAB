import matplotlib.pyplot as plt

# 定义文本框 和 箭头格式 【 sawtooth 波浪方框, round4 矩形方框 , fc表示字体颜色的深浅 0.1~0.9 依次变浅，没错是变浅】
decisionNode = dict(boxstyle="square", pad=0.5,fc="0.8")
leafNode = dict(boxstyle="circle", fc="0.8")
arrow_args = dict(arrowstyle="<-")
# 控制显示中文
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是否为dict, 不是+1
        if type(secondDict[key]) is dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是不是dict, 求分枝的深度
        # ----------写法1 start ---------------
        if type(secondDict[key]) is dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # ----------写法1 end ---------------

        # ----------写法2 start --------------
        # thisDepth = 1 + getTreeDepth(secondDict[key]) if type(secondDict[key]) is dict else 1
        # ----------写法2 end --------------
        # 记录最大的分支深度
        maxDepth = max(maxDepth, thisDepth)
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    # 获取叶子节点的数量
    numLeafs = getNumLeafs(myTree)
    # 获取树的深度
    # depth = getTreeDepth(myTree)

    # 找出第1个中心点的位置，然后与 parentPt定点进行划线
    cntrPt = (plotTree.xOff + (1 + numLeafs) / 2 / plotTree.totalW, plotTree.yOff)
    # print(cntrPt)
    # 并打印输入对应的文字
    plotMidText(cntrPt, parentPt, nodeTxt)

    firstStr = list(myTree.keys())[0]
    # 可视化Node分支点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 根节点的值
    secondDict = myTree[firstStr]
    # y值 = 最高点-层数的高度[第二个节点位置]
    plotTree.yOff = plotTree.yOff - 1 / plotTree.totalD
    for key in secondDict.keys():
        # 判断该节点是否是Node节点
        if type(secondDict[key]) is dict:
            # 如果是就递归调用[recursion]
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果不是，就在原来节点一半的地方找到节点的坐标
            plotTree.xOff = plotTree.xOff + 1 / plotTree.totalW
            # 可视化该节点位置
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 并打印输入对应的文字
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD


def createPlot(inTree):
    # 创建一个figure的模版
    fig = plt.figure(1, facecolor='green')
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    # 表示创建一个1行，1列的图，createPlot.ax1 为第 1 个子图，
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 半个节点的长度
    plotTree.xOff = -0.1 / plotTree.totalW
    plotTree.yOff = 0.5
    plotTree(inTree, (0.5, 0.5), '')
    plt.show()

import math
import pandas as pd
import numpy as np

def entropy(data):
    label_values = data[data.columns[-1]]
    #Returns object containing counts of unique values.
    counts =  label_values.value_counts()
    s = 0
    for c in label_values.unique():
        freq = float(counts[c])/len(label_values) 
        s -= freq*math.log(freq,2)
    return s

def is_continuous(data,attr):
    """Check if attr is a continuous attribute"""
    return data[attr].dtype == 'float64'

def split_points(data,attr):
    """Returns Ta,Equation(4.7),p.84"""
    values = np.sort(data[attr].values)
    return [(x+y)/2 for x,y in zip(values[:-1],values[1:])] 
def discrete_gain(data,attr):
    V = data[attr].unique()
    s = 0
    for v in V:
        data_v = data[data[attr]== v]
        s += float(len(data_v))/len(data)*entropy(data_v)
    return (entropy(data) - s,None)

def continuous_gain(data,attr,points):
    """Equation(4.8),p.84,returns the max gain along with its splitting point"""
    entD = entropy(data)
    #gains is a list of pairs of the form (gain,t)
    gains = []
    for t in points:
        d_plus = data[data[attr] > t]
        d_minus = data[data[attr] <= t]
        gain = entD - (float(len(d_plus))/len(data)*entropy(d_plus)+float(len(d_minus))/len(data)*entropy(d_minus))
        gains.append((gain,t))
    return max(gains)
def gain(data,attr):
    if is_continuous(data,attr):
        points = split_points(data,attr)
        return continuous_gain(data,attr,points)
    else:
        return discrete_gain(data,attr)
def majority(label_values):
    counts = label_values.value_counts()
    return counts.index[0]
def id3(data):
    attrs = data.columns[:-1]
    #attrWithGain is of the form [(attr,(gain,t))], t is None if attr is discrete
    attrWithGain = [(a,gain(data,a)) for a in attrs] 
    attrWithGain.sort(key = lambda tup:tup[1][0],reverse = True)
    return attrWithGain[0]
def createTree(data,split_function):
    label = data.columns[-1]
    label_values = data[label]
    #Stop when all classes are equal
    if len(label_values.unique()) == 1:
        return label_values.values[0]
    #When no more features, or only one feature with same values, return majority
    if data.shape[1] == 1 or (data.shape[1]==2 and len(data.T.ix[0].unique())==1):
        return majority(label_values)
    bestAttr,(g,t) = split_function(data)
    #If bestAttr is discrete
    if t is None:
        #In this tree,a key is a node, the value is a list of trees,also a dictionary
        myTree = {bestAttr:{}}
        values = data[bestAttr].unique() 
        for v in values:
            data_v = data[data[bestAttr]== v]
            attrsAndLabel = data.columns.tolist()
            attrsAndLabel.remove(bestAttr)
            data_v = data_v[attrsAndLabel]
            myTree[bestAttr][v] = createTree(data_v,split_function)
        return myTree
    #If bestAttr is continuous
    else:
        t = round(t,3)
        node = bestAttr+'<='+str(t)
        myTree = {node:{}}
        values = ['yes','no']
        for v in values:
            data_v = data[data[bestAttr] <= t] if v == 'yes' else data[data[bestAttr] > t]
            myTree[node][v] = createTree(data_v,split_function)
        return myTree
    
if __name__ == "__main__":
    f = pd.read_csv(filepath_or_buffer = 'watermelon3.0en.csv', sep = ',')
    data = f[f.columns[1:]]

    tree = createTree(data,id3)
    print(tree)
    createPlot(tree)