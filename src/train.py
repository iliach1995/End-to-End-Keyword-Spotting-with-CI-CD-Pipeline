"""
@author: ilia chiniforooshan

Script to perform model training.
"""

from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from src.experiment_tracking import MLFlowTracker
from src.model_train import CNN
from dataclasses import dataclass
import numpy as np

@dataclass
class Training():

    """
    Instance variables
    ------------------
    model: CNN, Instance of CNN class (in model_train) holding the created model.
    dataset: np.ndarray, the dataset where training happened
    labels: np.ndarray, the labels of the dataset
    validation_dataset: tuple of (validation data, validation labels)
    batch_size: int, Number of samples per gradient update.
    epochs: int, Number of epochs to train the model.
    learning_rate: float, Rate of model training. 
    optimizer: Adam, the name of the optimizer
    tracker: MLFlowTracker, Instance of MLFlowTracker class.
    metric_name: str, Metric name to sort the models.

    Returns
    -------
        None.
    """

    model: Sequential
    dataset: np.ndarray
    labels: np.ndarray
    validation_dataset: tuple
    batch_size: int
    epochs: int
    learning_rate: float
    tracker: MLFlowTracker
    metric_name: str

    def train_model(self)-> None:
        """
        Method that initializes and performs training.

        Parameters
        ----------
        None
        
        Returns
        -----------
        None

        Raises
        ------
        ValueError: Exception, if self.metric_name is not given or null.
        """

        if self.metric_name is None:
            raise ValueError("Please provide the metric name for model selection !!!")

        print("Training started.....")

        self.model.compile(optimizer= optimizers.Adam(learning_rate = self.learning_rate),
                           loss = 'categorical_crossentropy',
                           metrics=[self.metric_name])
        
        history = self.model.fit(
            x = self.dataset,
            y = self.labels,
            batch_size = self.batch_size,
            epochs = self.epochs,
            verbose = 1,
            validation_data= self.validation_dataset
        )

        return None



