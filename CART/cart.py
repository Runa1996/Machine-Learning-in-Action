import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fitline = list(map(float, curline))
        dataMat.append(fitline)
    return dataMat

def plotDataSet(filename):
    dataMat = loadDataSet(filename)
    n = len(dataMat)
    xcord = np.array(dataMat)[:,0].tolist()
    ycord = np.array(dataMat)[:,1].tolist()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s = 20, c = 'blue', alpha = .5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def plotDataSet2(filename):
    dataMat = loadDataSet(filename)                                        #加载数据集
    n = len(dataMat)                                                    #数据个数
    xcord = []; ycord = []                                                #样本点
    for i in range(n):                                                    
        xcord.append(dataMat[i][1]); ycord.append(dataMat[i][2])        #样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)                #绘制样本点
    plt.title('DataSet')                                                #绘制title
    plt.xlabel('X')
    plt.show()

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType = regLeaf,errType = regErr, ops = (1,4)):
    import types
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN or (np.shape(mat1)[0] < tolN)): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    import types
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] == getMean(tree['right'])
    if isTree(tree['left']): tree['left'] == getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree)
    #如果有左子树或者右子树,则切分数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    #处理左子树(剪枝)
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    #处理右子树(剪枝)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #如果当前结点的左右结点为叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) + np.sum(np.power(rSet[:,-1] - tree['right'],2))
        #计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        #计算合并的误差
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean, 2))
        #如果合并的误差小于没有合并的误差,则合并
        if errorMerge < errorNoMerge:
            return treeMean
        else: return tree
    else: return tree

if __name__ == "__main__":
    train_filename = 'ex2.txt'
    train_Data = loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = createTree(train_Mat)
    print('Before:\n',tree)
    test_filename = 'ex2test.txt'
    test_Data = loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    print('After:\n',prune(tree, test_Mat))