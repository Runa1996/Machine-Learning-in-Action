import numpy as np
from functools import reduce
import re
import random

def loadDataset():
    posting = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],['stop', 'posting', 'stupid', 'worthless', 'garbage'],['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return posting, classVec

def textParse(words):
    listOftokens = re.split(r'\W*', words)
    return [tok.lower() for tok in listOftokens if len(tok) > 2]

def createVocabList(Dataset):
    vocabSet = set([])
    for document in Dataset:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print('{} is not in Vocabulary!'.format(word))
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

def classify(vec2classify, p0vec, p1vec, pclasss1):
    p1 = sum(vec2classify * p1vec) + np.log(pclasss1)
    p0 = sum(vec2classify * p0vec) + np.log(1.0 - pclasss1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    Posting, Classlist = loadDataset()
    myVocabList = createVocabList(Posting)
    trainMat = []
    for pos in Posting:
        trainMat.append(setOfWords2Vec(myVocabList, pos))
    p0v, p1v, pAb = trainNB(np.array(trainMat),np.array(Classlist))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classify(thisDoc, p0v, p1v, pAb):
        print(testEntry,' is dirty word')
    else:
        print(testEntry, 'is not dirty word')
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classify(thisDoc, p0v, p1v, pAb):
        print(testEntry,' is dirty word')
    else:
        print(testEntry, 'is not dirty word')

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/{}.txt'.format(i),'r').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/{}.txt'.format(i),'r').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 取10个作为测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classify(np.array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == "__main__":
    spamTest()