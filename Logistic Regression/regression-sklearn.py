from sklearn.linear_model import LogisticRegression

def colicSklearn():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainLabel = []
    testSet = []
    testLabel = []
    for line in frTrain:
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainLabel.append(float(currLine[-1]))
    for line in frTest:
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabel.append(float(currLine[-1]))
    classifier = LogisticRegression(solver = 'liblinear', max_iter = 5000).fit(trainingSet,trainLabel)
    test_accuracy = classifier.score(testSet,testLabel) * 100
    print('True rate: {:.2f}'.format(test_accuracy))

if __name__ == "__main__":
    colicSklearn()