"""
@author: ilia chiniforooshan

This class determines configurations for model and data
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    epochs: int = 25
    learning_rate: float = 0.001
    num_filters: list = [32, 64, 128, 256]
    num_dense_layer: list = [512, 256]
    kernel_size: tuple = (3,3)
    pool_size: tuple = (2,2)
    dropout: float = 0.25
    num_labels: int = 31

@dataclass
class DataProcessConfig:
    mfcc_length: int = 40
    sampling_rate: int = 16000
    n_mfcc: int = 99
    batch_size: int
    