import numpy as np
import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def createDataset():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']             #分类属性
    return dataSet, labels

def calShannonEnt(dataSet):
    dataLength = len(dataSet)
    labels = np.array(dataSet)
    count = Counter(labels[:,-1].tolist())
    shannonEnt = 0.0
    for value in count.values():
        shannonEnt += -value/dataLength * np.log2(value/dataLength)
    return shannonEnt

def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1   # 得到属性的种类，遍历每个列
    totalfeatures = len(dataSet)
    baseEntropy = calShannonEnt(dataSet)    # 得到基础的香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    Labels = set(np.array(dataSet)[:,-1].tolist()) # 得到类的种类个数
    pdata = pd.DataFrame(dataSet)  # 把data转换成一个Dataframe
    for i in range(numFeatures):    # 遍历每一列的属性
        feature = set(np.array(dataSet)[:,i].tolist()) # 得到每一列属性a的个数
        partEntropy = 0.0 # 每个属性的经验条件熵
        for f in feature:   # 遍历属性的不同种类
            features = pdata[pdata.iloc[:,i] == int(f)].iloc[:,-1].tolist()   #得到该属性对应的行所对应的类
            partfeatures = len(features)    # 得到该属性的个数
            featCounts = Counter(features)
            for value in featCounts.values():
                partEntropy += -partfeatures / totalfeatures * (value / partfeatures) * np.log2(value / partfeatures)   # 得到经验条件熵
        infoGain = baseEntropy - partEntropy    # 得到信息增益的值
        # print("{}th infogain is:{:.3f}".format(i,infoGain))
        if infoGain > bestInfoGain: # 获取最大信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def createTree(dataSet, labels ,featLabels):
    classList = np.array(dataSet)[:,-1].tolist()
    # 当最后一列类只有一种的时候，直接返回类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 当前的数据D全部属于同一个类或者已经没有属性可以用来划分的时候，选择种类数量最大的一个作为子节点
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return Counter(classList).most_common[0][0]
    bestFeat = chooseBestFeature(dataSet)   # 选信息增益最大的分割点
    bestLabel = labels[bestFeat]    # 对应的属性名
    featLabels.append(bestLabel)
    myTree = {bestLabel:{}}
    del(labels[bestFeat])   # 排除该属性，继续划分
    featValues = set(np.array(dataSet)[:,int(bestFeat)].tolist())   # 对最好属性的每个取值后分类的内容做划分
    for value in featValues:
        Data = pd.DataFrame(dataSet)
        TempData = Data[Data.iloc[:,bestFeat] == int(value)].drop(bestFeat, axis = 1).values
        myTree[bestLabel][int(value)] = createTree(TempData, labels, featLabels)
    return myTree
    

def getNumLeafs(myTree):
    numLeafs = 0                                                #初始化叶子
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0                                                #初始化决策树深度
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth            #更新层数
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)        #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                            #计算标注位置                   
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        #设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)                                                          #获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                                                            #获取决策树层数
    firstStr = next(iter(myTree))                                                            #下个字典                                                 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    #标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():                               
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值                                             
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                                                    #创建fig
    fig.clf()                                                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                                            #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                                            #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                #x偏移
    plotTree(inTree, (0.5,1.0), '')                                                            #绘制决策树
    plt.show()                                                                                 #显示绘制结果

def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    #print(firstStr)
    secondDict = inputTree[firstStr]
    #print(secondDict)
    featIndex = featLabels.index(firstStr)
    #print(featIndex)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    with open(filename, 'wb') as f:
        pickle.dump(inputTree,f)

def loadTree(filename):
    f = open(filename, 'rb')
    return pickle.load(f)

if __name__ == '__main__':
    myTree = loadTree('classifierTree.txt')
    print(myTree)