import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import random

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./testSet.txt')
    for line in fr:
        lineArr = line.strip().split('\t')
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# def gradAscent(dataMat, classLabels):
#     dataMatrix = np.mat(dataMat)
#     labelMat = np.mat(classLabels).transpose()
#     m, n = np.shape(dataMatrix)
#     alpha = 0.001
#     maxCycles = 500
#     weights = np.ones((n,1))    # n x 1
#     weights_array = np.array([])
#     for i in range(maxCycles):
#         h = sigmoid(dataMatrix * weights)   # m x n x n x 1 = m x 1
#         error = labelMat - h    # m x 1
#         weights = weights + alpha * dataMatrix.transpose() * error  # n x 1 + n x m x m x 1 = n x 1
#         weights_array = np.append(weights_array,weights)
#     weights_array = weights_array.reshape(maxCycles,n)
#     return weights.getA(), weights_array

def gradAscent(dataMat, classLabels):
    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))    # n x 1
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)   # m x n x n x 1 = m x 1
        error = labelMat - h    # m x 1
        weights = weights + alpha * dataMatrix.transpose() * error  # n x 1 + n x m x m x 1 = n x 1
    return weights.getA()

def stocGradAscent(dataMat, classLabels, numIter = 150):
    m, n = np.shape(dataMat)    # m x n
    weights = np.ones(n)    # 1 x n
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))  # 1 x n
            error = classLabels[randIndex] - h  # 1 x 1
            weights = weights + alpha * error * dataMat[randIndex]  # 1 x n + 1 x n
            del(dataIndex[randIndex])
    return weights

# def stocGradAscent(dataMat, classLabels, numIter = 150):
#     m, n = np.shape(dataMat)    # m x n
#     weights = np.ones(n)    # 1 x n
#     weights_array = np.array([])
#     for j in range(numIter):
#         dataIndex = list(range(m))
#         for i in range(m):
#             alpha = 4 / (1.0 + j + i) + 0.01
#             randIndex = int(random.uniform(0,len(dataIndex)))
#             h = sigmoid(sum(dataMat[randIndex] * weights))  # 1 x n
#             error = classLabels[randIndex] - h  # 1 x 1
#             weights = weights + alpha * error * dataMat[randIndex]  # 1 x n + 1 x n
#             weights_array = np.append(weights_array, weights, axis = 0)
#             del(dataIndex[randIndex])
#     weights_array = weights_array.reshape(numIter * m, n)
#     return weights, weights_array

def colicTest(grad='gd'):
    frTrain = open('horseColicTraining.txt','r')
    frTest = open('horseColicTest.txt','r')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    if grad == 'stoc':
        trainingWeights = stocGradAscent(np.array(trainingSet), trainingLabels, 500)
    else:
        trainingWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if grad == 'stoc':
            if int(classifyVector(np.array(lineArr),trainingWeights)) != int(currLine[-1]):
                errorCount += 1
        else:
            if int(classifyVector(np.array(lineArr),trainingWeights[:,0])) != int(currLine[-1]):
                errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100
    print('Error Rate in Test is {}'.format(errorRate))

def classifyVector(dataMat, weights):
    prob = sigmoid(sum(dataMat * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def plotDataSet():
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's', alpha = .5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha = .5)
    plt.title('Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()

def plotWeights(weights_array1,weights_array2):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(20,10))
    x1 = np.arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()

if __name__ == "__main__":
    # dataMat, classLabel = loadDataSet()
    # weights1,weights_array1 = stocGradAscent(np.array(dataMat), classLabel)

    # weights2,weights_array2 = gradAscent(dataMat, classLabel)
    # plotWeights(weights_array1, weights_array2)

    colicTest()