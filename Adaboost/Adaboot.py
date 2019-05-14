import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def loadSimpData():
    
    dataMat = np.matrix([[ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMat)[0],1))
    if threshIneq == 'lt':
        retArray[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, ClassLabel, D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = np.mat(dataArr); labelMat = np.mat(ClassLabel).T
    m,n = np.shape(dataMatrix)
    numSteps = 10
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin+float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:     #找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    '''
    adaboost算法过程
    '''
    weakClassArr = []   # weakClassArr保存每次迭代产生的最优基本分类器
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)    # 最开始将所有权重初始化为1/m
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)    # 将初始化的内容输入，获取第一个基本分类器
        #print('D:', D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 计算G(x)的系数 α =1/2*log(1-e/e)
        bestStump['alpha'] = alpha  # 记录α的值
        weakClassArr.append(bestStump)  # 记录第一次基本分类器
        #print('classEst:',classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)   # 计算上标exp中的内容(-αm*yi*G(x))
        D = np.multiply(D, np.exp(expon))   # 计算Wm*exp(expon)
        D = D / D.sum() # 计算新的权值分部D/Zm
        aggClassEst += alpha * classEst # 构建线性组合f(x)=α*G(x)
        #print('aggClassEst:',aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))  # 使用sign函数得到结果和真实的label进行比较，得到错误的项
        errorRate = aggErrors.sum() / m # 计算错误率
        #print('total error:',errorRate)
        if errorRate == 0.0: break      # 如果错误率为0，则完成构建
    return weakClassArr, aggClassEst

def adaClassify(dataToClass, classifierArr):
    '''
    AdaBoost分类函数
    Parameters:
        dataToClass 待分类的样例
        classifierArr 训练好的分类器
    '''
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print(aggClassEst)
    return np.sign(aggClassEst)
    

def plotROC(predStrengths, classLabels):
    """
    绘制ROC
    Parameters:
        predStrengths - 分类器的预测强度
        classLabels - 类别
    Returns:
        无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    cur = (1.0, 1.0)                                                         #绘制光标的位置
    ySum = 0.0                                                                 #用于计算AUC
    numPosClas = np.sum(np.array(classLabels) == 1.0)                        #统计正类的数量
    yStep = 1 / float(numPosClas)                                             #y轴步长   
    xStep = 1 / float(len(classLabels) - numPosClas)                         #x轴步长
 
    sortedIndicies = predStrengths.argsort()                                 #预测强度排序
 
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]                                                     #高度累加
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')     #绘制ROC
        cur = (cur[0] - delX, cur[1] - delY)                                 #更新绘制光标的位置
    ax.plot([0,1], [0,1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties = font)
    plt.xlabel('假阳率', FontProperties = font)
    plt.ylabel('真阳率', FontProperties = font)
    ax.axis([0, 1, 0, 1])
    print('AUC面积为:', ySum * xStep)                                         #计算AUC
    plt.show()

def showDataSet(dataMat, labelMat):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    print(data_plus_np.shape,data_minus_np.shape)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr, 10)
    plotROC(aggClassEst.T, LabelArr)