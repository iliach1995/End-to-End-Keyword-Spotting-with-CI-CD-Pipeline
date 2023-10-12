"""
@author ilia chiniforooshan esfahani
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

def loadData(path: str , dataType: str) -> (np.ndarray, np.ndarray):
    
    """
    This function load the saved numpy array files of audios

    Parameters
    ----------------------
    path: str, load the saved data from .npy file
    dataTpoe: str, the type of loading -- train, val, dev, test
    
    Return
    ---------------------
    data: np.ndarray, the loaded data
    labels: np.ndarray, the corresponding labels
    """
    data = np.load(path +"/" + dataType + ".npy")
    labels = np.load(path +"/" + dataType + "Labels" + ".npy")

    return data, labels

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
    trainData, trainLabels = loadData(path = "./inputnpy/main", dataType='train')
    visualizeClassBalance(trainLabels)

    valData, valLabels = loadData(path = "./inputnpy/main", dataType='val')
    visualizeClassBalance(valLabels)

    devData, devLabels = loadData(path = "./inputnpy/main", dataType='dev')
    visualizeClassBalance(devLabels)

    testData, testLabels = loadData(path = "./inputnpy/main", dataType='test')
    visualizeClassBalance(testLabels)