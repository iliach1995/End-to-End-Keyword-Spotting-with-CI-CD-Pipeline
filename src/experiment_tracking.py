"""
@author: ilia esfahani

Tracks model training and log the model artifacts along with resulting metrics
and parameters. For that purpose, `MLFlow` is used. It has the flexibility to
extend its functionality to support other tracking mechanism like tensorboard etc.
"""

import mlflow
import pandas as pd
from typing import Protocol
from dataclasses import dataclass, field


@ dataclass
class MLFlowTracker:
    """
    class to track experiments using MLFlow Tracker. 

    Parameters
    ------------------
    experiment_name: str, name of experiment to be activated
    tracking_uri: str, An HTTP URI or local file path, prefixed with `file:/`

    Returns
    -------
    None
    """
    experiment_name: str
    tracking_uri: str

    def __start__(self)-> None:
        """
        Dunder method, sets tracking URI and experiment name to MLFlow engine.
        """
        mlflow.set_experiment = self.experiment_name
        mlflow.set_tracking_uri = self.tracking_uri
    
    def log(self) -> None:
        
        """
        Auto logging for tracking experiment.
        log model artifacts, parameters and metrics in the ./artifacts directory.
        """
        self.__start__()
        mlflow.keras.autolog()

    def find_best_model(self, metric: str)-> None:
        """
        This is the method for model selection. Find the best model by sorting
        the model using the given metric.

        PS. this can be used using mlflow ui, this is the code implementation.

        Parameters
        --------------
        metric: str, metric name to sort the models.

        Returns
        -------
        None

        Raises
        ------
        MLFlowError: Exception, if the experiment id or 
            experiment name is none/invalid.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        experiment_id = experiment.experiment_id

        if experiment is None or experiment_id is None:
            raise Exception(
                f"MLFlow Error, Invalid experiment details.Please re-check them and try again !!!")

        result_df = mlflow.search_runs([experiment_id], 
                                        order_by=[f"metrics.{metric} DESC"])

        return None