"""
@author ilia chiniforooshan 
load .npy data and visualize the train, val, and test class (bar plot)
This indicates that we have unimablanced dataset or not.
This imabanace dataset should be reflected in train, val, test in the same trend
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from parameters import *
import collections
import matplotlib.pyplot as plt
from src.load_data import load_data

def visualizeClassBalance(labels: np.ndarray) -> None:

    """
    this function is used to visualize the number of each label in dataset

    Parameters
    --------------
    labels: np.ndarray, the labels of the audios

    Return
    -------------
    None
    """
    numberLabels = []
    
    counter = collections.Counter(labels)
    x = list(counter.keys())
    y = list(counter.values())

    yBetween =np.where((np.abs(np.array(y) - 3000)<=1000), y, np.nan)
    yBetween = yBetween[~np.isnan(yBetween)]
    yAvg = np.mean(yBetween)

    weights = yAvg/y
    plt.bar(x = x, height = y)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()




if __name__ == '__main__':
    trainData, trainLabels = load_data(path = "./inputnpy/main", dataType='train')
    visualizeClassBalance(trainLabels)

    valData, valLabels = load_data(path = "./inputnpy/main", dataType='val')
    visualizeClassBalance(valLabels)

    devData, devLabels = load_data(path = "./inputnpy/main", dataType='dev')
    visualizeClassBalance(devLabels)

    testData, testLabels = load_data(path = "./inputnpy/main", dataType='test')
    visualizeClassBalance(testLabels)