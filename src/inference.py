"""
@author: ilia chiniforooshan

Script for inferencing with new or test data. It implements functionality
to make predictions on the real data from trained model artifact.
"""

import os
import mlflow
import numpy as np
from typing import Any, Tuple
import mlflow
from mlflow import keras
from keras.models import Sequential
from src.audio_process import AudioProcess
from src.parameters import inv_categories

class KeywordSpotter:
    
    def __init__(self, model_artifactory_dir: str,
                n_mfcc: int, mfcc_length: int, sampling_rate: int) -> None:
        """
        Parameters
        ----------
        model_artifactory_dir: str, Directory that holds trained model artifacts.
        n_mfcc: int, Number of MFCCs to return.
        mfcc_length: int, Length of MFCC features for each audio input.
        sampling_rate: int, Target sampling rate
        """
        self.model_artifactory_dir = model_artifactory_dir
        self.n_mfcc = n_mfcc
        self.mfcc_length = mfcc_length
        self.sampling_rate = sampling_rate
        self.model = None

    def load_model(self) -> None:
        """
        loading model from artifacts

        Parameters
        --------------
        None

        Return
        --------------
        None
        """

        if not os.path.exists(self.model_artifactory_dir):
                raise Exception(
                       f"{self.model_artifactory_dir} doesn't exists. Please enter a valid path !!!"
                        )
        
        model: Sequential = mlflow.keras.load_model(self.model_artifactory_dir)
        self.model = model


    def predict_from_audio(self, audio_file: str) -> Tuple[str, float]:
        """
        Method to make predictions based on probabilities from the model on the given
        audio file.

        Parameters
        ----------
        audio_file: str, path to audio file

        Return
        ------
        predicted_keyword: str, Predicted keyword from the model as text.
        label_probability: float, Probability of the predicted keyword.

        Raises
        ------
        ValueError: Exception, If predicted_keyword or label_probability is none.
        NotFoundError: Exception, When an exception is caught by the `try` block.
        """

        try:
            
            if self.model is None:
                self.load_model()


            audio_process = AudioProcess(sr = self.sampling_rate,
                                         nMfcc= self.n_mfcc,
                                         mfccLength= self.mfcc_length)

            audio_mfcc: np.ndarray = audio_process.audiotoMfccFile(audio_file)
            
            predicted_keyword, label_probability = self.predict_from_mfcc(audio_mfcc)
           
            return predicted_keyword, label_probability

        except Exception as exc:
            raise Exception (f"Cannot infer from model. Please check the paths and try it again !!! {exc}") from exc

    def predict_from_mfcc(self, mfcc: np.ndarray) -> Tuple[str, float]:
        
        """
        Method to make predictions based on probabilities from the model on the given
        mfcc file.

        Parameters
        ----------
        mfcc: np.ndarray, mfcc format (numpy array)

        Return
        ------
        predicted_keyword: str, Predicted keyword from the model as text.
        label_probability: float, Probability of the predicted keyword.

        Raises
        ------
        ValueError: Exception, If predicted_keyword or label_probability is none.
        NotFoundError: Exception, When an exception is caught by the `try` block.
        """
        try:
            
            if self.model is None:
                self.load_model()

            if mfcc.ndim == 2:
                mfcc = np.reshape(mfcc, (1,mfcc.shape[0],mfcc.shape[1]))
            model_output = self.model.predict(mfcc)
            predicted_keyword: str = inv_categories.get(np.argmax(model_output))
            label_probability: float = max([round(value,4) for value in 
                                list(dict(enumerate(model_output.flatten(), 1)).values())])
           
            if predicted_keyword is None or label_probability is None :
                raise ValueError("Model returned empty predictions!!!")
           
            return predicted_keyword, label_probability
        
        except Exception as exc:
            raise Exception (f"Cannot infer from model. Please check the paths and try it again !!! {exc}") from exc


if __name__ == '__main__':
    kws = KeywordSpotter("./artifacts/model", 99, 40, 16000)
    predicted_keyword, label_probability = kws.predict_from_audio("./dataset/test/audio_test.wav")

