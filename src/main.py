"""
@author: ilia chiniforooshan
Script to train the model using .npy file using CNN model 
and logged model artifacts using mlflow
"""
import warnings
from keras.utils import to_categorical
from keras.models import Sequential
from src.experiment_tracking import MLFlowTracker
#from src.model_train import CNN
from src.data_preparation import Dataset, PreProcess
from src.load_data import load_data
from src import train
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from config.config_type import ModelConfig, DataProcessConfig

from src.model_train2 import CNN


warnings.filterwarnings('ignore')

def main() -> None:
    try:

        dataConfig = DataProcessConfig()
        modelConfig = ModelConfig()
        """
        if using for the first time, use the below code. if you have npy files
        use loading data code!


        dataset = Dataset()
        
        preProcess = PreProcess(dataset=dataset, 
                                sr = dataConfig.sampling_rate,
                                nMfcc = dataConfig.n_mfcc,
                                mfccLength= dataConfig.mfcc_length)
        
        dataset_train, labels_train = preProcess.audio_files_to_numpy(fileType='train')
        dataset_val, labels_val = preProcess.audio_files_to_numpy(fileType='train')
        """
        

        #When Loading Data, use only this part


        dataset_train, labels_train = load_data(path = "./inputnpy", dataType='train')
        dataset_val, labels_val = load_data(path = "./inputnpy", dataType='val')


        labels_train_encoded = to_categorical(labels_train)
        labels_val_encoded = to_categorical(labels_val)
        
        
        tracker = MLFlowTracker(experiment_name='CNN model', tracking_uri='./artifacts')
        now = datetime.now()
        dt_string = now.strftime("%m-%d-%Y-%H-%M-%S")
        tracker.log(dt_string)

        
        model: Sequential = CNN(num_filters = modelConfig.num_filters,
                                num_dense_layer= modelConfig.num_dense_layer,
                                input_shape=(dataConfig.n_mfcc, dataConfig.mfcc_length),
                                target_shape=(dataConfig.n_mfcc,dataConfig.mfcc_length,1),
                                kernel_size=modelConfig.kernel_size,
                                pool_size=modelConfig.pool_size,
                                dropout=modelConfig.dropout,
                                num_labels=31).transfer_learning()

        
        train.Training( model = model,
                       dataset = dataset_train,
                       labels = labels_train_encoded,
                       validation_dataset=(dataset_val, labels_val_encoded),
                       batch_size=128,
                       epochs = 25,
                       learning_rate=0.001,
                       tracker = tracker,
                       metric_name = 'categorical_accuracy').train_model()



    except Exception as exc:
        raise Exception("Problem in runnig model") from exc
    

if __name__ == "__main__":
    main()