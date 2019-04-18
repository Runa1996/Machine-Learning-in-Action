import numpy as np
import pandas as pd
from collections import Counter

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
    labels = ['不放贷', '放贷']             #分类属性
    return dataSet, labels

def calShannonEnt(dataSet):
    dataLength = len(dataSet)
    labels = np.array(dataSet)
    count = Counter(labels[:,-1].tolist())
    shannonEnt = 0.0
    for value in count.values():
        shannonEnt += -value/dataLength * np.log2(value/dataLength)
    return shannonEnt

def BestFeatureToSpilt(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    Labels = set(np.array(data)[:,-1].tolist())
    pdata = pd.Dataframe(data)
    for i in range(numFeatures):
        feature = set(np.array(data)[:,i].tolist())
        for f in feature:
            features = pdata[pdata.iloc[:,i] == f].iloc[:-1]
            featCounts = Counter(features.tolist())
            
    

if __name__ == "__main__":
    data, features = createDataset()
    print(calShannonEnt(data))
