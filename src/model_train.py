"""
@author: ilia chiniforooshan
Buulding model using our CNN class

"""
from keras.applications.vgg19 import VGG19, preprocess_input
import mlflow
import mlflow.keras
from dataclasses import dataclass
from keras.models import Model
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Reshape, BatchNormalization, Dropout, ReLU
from parameters import *
import mlflow.tensorflow as tf
from src.load_data import load_data


@dataclass
class CNN():

    """
    create a CNN class for building a model
    
    Instances:
    ----------------
    num_filters: list. a list contatining the number of filters for each conv2d 
    num_dense_layer: list, contaiing number of nodes for each layer
    input_shape: the input shape of the model for reshaping layer (first layer)
    target_shape: the output shape of the model for reshaping layer (first layer)
    kernel_size: tuple include the size of kernel for conv2d layers
    pool_size: pool size for maxpolling 2d layer
    DROPOUT: float, amount of drop out after eacl layer
    num_labels: int, the number of labels/categories
    """
    num_filters: list
    num_dense_layer: list
    input_shape: tuple
    target_shape: tuple
    kernel_size: tuple
    pool_size: tuple
    dropout: float
    num_labels: int


    def build_model(self) -> Sequential:
        
        """
        Method to define model that can be used for training
        and inference. This existing model can also be tweaked
        by changing parameters, based on the requirements.

        

        Return
        ---------------
        Sequential model
        """

        model = Sequential()
        model.add(Reshape(input_shape=self.input_shape, target_shape=self.target_shape))
        model.add(BatchNormalization())
        model.add(ReLU())

        for filter in self.num_filters:
            model.add(Conv2D(filters = filter, kernel_size=self.kernel_size, padding="same"))
            model.add(BatchNormalization())
            model.add(ReLU())
            model.add(MaxPooling2D(pool_size=self.pool_size))
            model.add(Dropout(self.dropout))
        
        model.add(Flatten())

        count = 1
        for dense_layer in self.num_dense_layer:
            model.add(Dense(dense_layer, name= 'layer'+ str(count)))
            model.add(ReLU())
            model.add(Dropout(self.dropout))
            count+=1

        model.add(Dense(self.num_labels, activation='softmax'))

        return model

    
    def tranfer_learning_VGG19(self, fine_tune: int = 3) -> Sequential:
        """
        method for making transfer learning model using VGG 16

        Parameters
        --------------
        fine_tune: int, the number of last layer in Conv/pooling for fine tuning

        Return
        ----------------
        model: Sequential
        """

        conv_base = VGG19(include_top=False,
                     weights='imagenet', 
                     input_shape=self.input_shape)
        
        if fine_tune > 0:
            for layer in conv_base.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        top_model = conv_base.output

        top_model = Flatten(name="flatten")(top_model)

        count = 1
        for dense_layer in self.num_dense_layer:
            top_model = Dense(dense_layer, name= 'layer'+ str(count))(top_model)
            top_model = BatchNormalization()(top_model)
            top_model = ReLU()(top_model)
            top_model = Dropout(self.dropout)(top_model)
            count+=1

        output_layer = Dense(self.num_labels, activation='softmax')(top_model)
        
        model = Model(inputs=conv_base.input, outputs=output_layer)

        return model


if __name__ == "__main__":
    trainData, trainLabels = load_data(path = "./inputnpy", dataType='train')
    if trainData.shape[0] > 1000:
        trainDataTruncate = trainData[0:1000,:,:]
    
    np.save("./inputnpy/train_truncate.npy", trainDataTruncate)
    cnn = CNN
    model = cnn.build_model()