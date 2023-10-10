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

    #a = 2
    return None




if __name__ == '__main__':
    trainData, trainLabels = loadData(path = "./inputnpy", dataType='train')
    visualizeClassBalance(trainLabels)