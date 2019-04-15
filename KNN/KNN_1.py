import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as knn

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    with open(filename, 'r' , encoding='utf-8') as f:
        for i in range(32):
            lines = f.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lines[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFilelist = listdir('./trainingDigits')
    m = len(trainingFilelist)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileName = trainingFilelist[i]
        classNumber = int(fileName.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/{}'.format(fileName))
    neigh = knn(n_neighbors=3, algorithm='auto')
    neigh.fit(trainingMat,hwLabels)
    testFileList = listdir('./testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileName = testFileList[i]
        classNumber = int(fileName.split('_')[0])
        vector = img2vector('./testDigits/{}'.format(fileName))
        classifierResult = neigh.predict(vector)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))
    
if __name__ == "__main__":
    handwritingClassTest()