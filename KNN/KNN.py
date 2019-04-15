import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split

def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

def file2Matrix(file):
    with open(file, 'r', encoding ='utf-8') as f:
        flines = f.readlines()
        NumberofLines = len(flines)
        returnMat = np.zeros((NumberofLines,3))
        classLabel = []
        index = 0
        for line in flines:
            line = line.strip()
            listForm = line.split('\t')
            returnMat[index,:] = listForm[0:3]
            if listForm[3] == 'didntLike':
                classLabel.append(1)
            elif listForm[3] == 'smallDoses':
                classLabel.append(2)
            elif listForm[3] == 'largeDoses':
                classLabel.append(3)
            index += 1
        return returnMat,classLabel

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    Scales = maxVals - minVals
    normDatasets = dataSet - np.tile(minVals,(dataSet.shape[0], 1))
    normDatasets = normDatasets / np.tile(Scales, (dataSet.shape[0], 1))
    return normDatasets, Scales, minVals

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize ,1)) - dataSet
    sqrtDiff = diffMat**2
    sqDistance = sqrtDiff.sum(axis = 1)
    distances = sqDistance ** 0.5
    sortDistance = distances.argsort()  # get index of sorted array
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortDistance[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortClassCount = sorted(classCount.items(),key = lambda item: item[1], reverse = True)
    
    return sortClassCount[0][0]

def classifyPersion():
    result = ['dislike', 'like', 'verylike']
    percentTats = float(input('Game percent:'))
    ffMiles = float(input('Fly miles every year:'))
    iceCream = float(input('IceCream eat every week:'))
    filename = './datingTestSet.txt'
    datingDataMat, datingLabels = file2Matrix(filename)
    normMat, ranges, minValue = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    normArr = (inArr - minValue) / ranges
    classifierResult = classify0(normArr, normMat, datingLabels, 3)
    print('Maybe you {} this man.'.format(result[classifierResult-1]))


def datingClassTest():
    filename='./datingTestSet.txt'
    datingDataMat, datingLabels = file2Matrix(filename)
    normData, ranges, minValue = autoNorm(datingDataMat)
    x_train,x_test,y_train,y_test = train_test_split(normData, datingLabels, test_size = 0.1)
    errorCount = 0
    for i in range(len(x_test)):
        classifierResult = classify0(x_test[i], x_train, y_train, 4)
        print('Result:{}, True Result:{}'.format(classifierResult,y_test[i]))
        if classifierResult != y_test[i]:
            errorCount += 1
    print('Error rate:{:.2f}%'.format(errorCount/float(len(x_test))*100))

if __name__ == "__main__":
    classifyPersion()