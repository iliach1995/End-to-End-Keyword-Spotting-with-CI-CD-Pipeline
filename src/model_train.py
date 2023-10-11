import mlflow
import mlflow.keras
from dataclasses import dataclass
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Reshape, BatchNormalization, Dropout, ReLU
from parameters import *
import mlflow.tensorflow as tf

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

@dataclass
class Classifier():
       
    def build_model(self, num_filters: list, num_dense_layer: list, 
                    INPUT_SHAPE: tuple, TARGET_SHAPE: tuple, 
                    KERNEL_SIZE, POOL_SIZE, DROPOUT: float) -> Sequential:

        model = Sequential()
        model.add(Reshape(input_shape=INPUT_SHAPE, target_shape=TARGET_SHAPE))
        model.add(BatchNormalization())
        model.add(ReLU())

        for filter in num_filters:
            model.add(Conv2D(filters = filter, kernel_size=KERNEL_SIZE, padding="same"))
            model.add(BatchNormalization())
            model.add(ReLU())
            model.add(MaxPooling2D(pool_size=POOL_SIZE))
            model.add(Dropout(DROPOUT))
        
        model.add(Flatten())

        count = 1
        for dense_layer in num_dense_layer:
            model.add(Dense(dense_layer, name= 'layer'+ str(count)))
            model.add(ReLU())
            model.add(Dropout(DROPOUT))

        model.add(Dense(NUM_LABELS))

    def compile(self, learning_rate: float, optimizer: float) -> None:

        self.optimizer = optimizer
        self.learning_rate = learning_rate

        return None




if __name__ == "__main__":
    trainData, trainLabels = loadData(path = "./inputnpy", dataType='train')
    if trainData.shape[0] > 1000:
        trainDataTruncate = trainData[0:1000,:,:]
    
    np.save("./inputnpy/train_truncate.npy", trainDataTruncate)
    slkjd = Classifier()
    slkjd.normalizeTrain()