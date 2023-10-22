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
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Layer
import librosa
from src.audio_process import AudioProcess

class MFCCLayer(Layer):
    def __init__(self,sr, n_mfcc, mfcc_length, **kwargs):
        super(MFCCLayer, self).__init__(**kwargs)
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.mfcc_length = mfcc_length

    def build(self, input_shape):
        super(MFCCLayer, self).build(input_shape)

    def call(self, inputs):

        audio_process = AudioProcess(sr = self.sr, nMfcc = self.n_mfcc, mfccLength=self.mfcc_length)
        mfcc = audio_process.audiotoMfccFile(inputs)
        return mfcc

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_mfcc, self.mfcc_length)

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

    def transfer_learning(self) -> Sequential:

        """
        Method to define model that can be used for training
        and inference. This existing model can also be tweaked
        by changing parameters, based on the requirements.

        

        Return
        ---------------
        Sequential model
        """
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        

        input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
        embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='yamnet')
        
        _, embeddings_output, _ = embedding_extraction_layer(input_segment)

        my_model = Sequential(name='custom_top')
        my_model.add(tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                                name='input_embedding'))
        
        count = 1
        for dense_layer in self.num_dense_layer:
            my_model.add(Dense(dense_layer, name= 'layer'+ str(count)))
            my_model.add(ReLU())
            my_model.add(Dropout(self.dropout))
            count+=1

        my_model.add(Dense(self.num_labels, activation='softmax'))
        
        serving_outputs = my_model(embeddings_output)

        serving_model = tf.keras.Model(input_segment, serving_outputs)
        for l in serving_model.layers:
            print(l.name, l.trainable)
        return serving_model
        


if __name__ == "__main__":
    trainData, trainLabels = load_data(path = "./inputnpy", dataType='train')
    if trainData.shape[0] > 1000:
        trainDataTruncate = trainData[0:1000,:,:]
    
    np.save("./inputnpy/train_truncate.npy", trainDataTruncate)
    cnn = CNN
    model = cnn.build_model()