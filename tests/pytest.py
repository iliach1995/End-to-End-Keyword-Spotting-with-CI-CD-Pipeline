"""
@author: ilia chiniforooshan

Script to perform unit testing using `pytest`.
"""

import pytest
import numpy as np
from src.parameters import categories
from config.config_type import DataProcessConfig
from src.audio_process import AudioProcess
from src.inference import KeywordSpotter
import os
@pytest.fixture
def mfcc() -> np.ndarray:
    """
    Fixture function to convert audio file to MFCC features to be
    used as a global variable in multiple tests.

    Parameters
    ----------
        None.

    Returns
    -------
    mfcc: np.ndarray
        Computed MFCC features
    """

    dataConfig = DataProcessConfig()
    audio_process = AudioProcess(sr = dataConfig.sampling_rate,
                                         nMfcc= dataConfig.n_mfcc,
                                         mfccLength= dataConfig.mfcc_length)
    mfcc_features = audio_process.audiotoMfccFile(path = "./dataset/test/audio_test.wav")
    return mfcc_features

def test_label_type() -> None:
    """Function to test the datatype of labels which should be
    `str` in order to be used for training and inference.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    """
    labels = categories.keys()
    assert all(isinstance(n, str) for n in labels)

def test_mfcc_shape(mfcc: pytest.fixture) -> None:
    """Function to test the shape of computed MFCC features
    from audio files. It is an ndarray whose shape should
    match the parameters(n_mfcc, mfcc_length) from the config.

    Parameters
    ----------
    mfcc: pytest.fixture, Computed MFCC features

    Returns
    -------
    None.
    """
    assert mfcc.shape == (DataProcessConfig.n_mfcc, DataProcessConfig.mfcc_length)

def test_mfcc_dimension(mfcc: pytest.fixture) -> None:
    """Function to test the dimension of mfcc features array.

    Parameters
    ----------
    mfcc: pytest.fixture
        Computed MFCC features

    Returns
    -------
        None.
    """
    assert len(mfcc.shape) == 2

def test_model_prediction() -> None:
    """
    test model prediction for the audio_test
    """
    dataConfig = DataProcessConfig()
    kws = KeywordSpotter("./artifacts/model", dataConfig.n_mfcc, dataConfig.mfcc_length, dataConfig.sampling_rate)

    allFiles = []
    for root, dirs, files in os.walk("./dataset/modeltest"):

        allFiles += [root + "/" + fileName for fileName in files if fileName.endswith(".wav")]

    count = 0
    for file in allFiles:
        predicted_keyword , _ = kws.predict_from_audio(file)

        category: str = file.split("/")[-1]
        category = category.replace('.wav', '')

        if category == predicted_keyword:
            count+=1
    
    assert count>3


