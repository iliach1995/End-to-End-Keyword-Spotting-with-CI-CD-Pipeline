"""
@author: ilia chiniforooshan
function for loading saved data (in .npy format)
"""

import numpy as np


def load_data(path: str , dataType: str) -> (np.ndarray, np.ndarray):
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