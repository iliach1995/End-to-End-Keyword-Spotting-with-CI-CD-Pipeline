"""
@author: ilia chiniforooshan

Dataset class for download the google speech dataset and Preprocess class for dividing train, validation
test dataset and process it to MFCC. Also, in preprocess, we save the audio files as .npy format for
future use which is much more easier.
"""
import tensorflow as tf
import os
import requests
import tarfile
from dataclasses import dataclass
import numpy as np
from scipy.io import wavfile
from parameters import *
import librosa
from tqdm import tqdm

@dataclass
class Dataset():

    def getData(self, inputPath: str = "./input/speech_commands"):

        """
        Download and Extract Google Speech Commands dataset (version 0.02)

        Parameters
        --------------
        inputPath: str, Path to download dataset

        Return
        --------
        None
        """

        datasets = ["train", "test"]

        urls = ["http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"]


        for dataset, url in zip(datasets, urls):
            datasetDirectory  = inputPath + "/" + dataset 

            if not os.path.isdir(datasetDirectory):
                os.makedirs(datasetDirectory)

            fileName = datasetDirectory + dataset + ".tar.gz"

            if not os.path.isfile(fileName):
                print(f'Downloading {dataset} dataset')

                dataRequest = requests.get(url)

                with open(fileName, "wb") as file:
                    file.write(dataRequest.content)
            
                if fileName.endswith("tar.gz"):
                    tar = tarfile.open(fileName, "r:gz")
                    tar.extractall(path=datasetDirectory)
                    tar.close()
        
        print("Data is Sucessfully Imported!")


    def getDataDict(self, inputPath: str = "./input/speech_commands")\
        -> dict: 

        """
        Prepare the training, validation, and testing files. 
        dumped them into dictionary mapping with train, test, validate and test file names and labels.

        Parameters
        --------------
        inputPath: str, Path to dataset

        Return
        --------
        None
        
        """

        # get validation files
        valDirectory =inputPath + "/train/" + "validation_list.txt"

        with open (valDirectory, 'r') as f:
            
            lines = f.read().split('\n')
            valFiles = [ inputPath + "/train/" + line for line in lines]
            valLabels = [self.getLabel(file) for file in valFiles]

        #get dev files
        devDirectory = inputPath + "/train/" + "testing_list.txt"
        
        with open (devDirectory, 'r') as f:

            lines = f.read().split('\n')
            devFiles = [ inputPath + "/train/" + line for line in lines]
            devLabels = [self.getLabel(file) for file in devFiles]

        # get all files (for train files)
        allFiles = []
        for root, dirs, files in os.walk(inputPath + "/train/"):

            allFiles += [root + "/" + fileName for fileName in files if fileName.endswith(".wav")]

        # build train files
        trainFiles = list(set(allFiles) - set(valFiles) - set(devFiles))
        trainLabels = [self.getLabel(file) for file in trainFiles]

        # get final test files
        allFiles = []
        for root, dirs, files in os.walk(inputPath + "/test/"):

            allFiles += [root + "/" + fileName for fileName in files if fileName.endswith(".wav")]

        testFiles = list(set(allFiles))    
        testLabels = [self.getLabel(file) for file in testFiles]

        # Create dictionaries containing (file, labels)
        trainData = {"files": trainFiles, "labels": trainLabels}
        valData = {"files": valFiles, "labels": valLabels}
        devData = {"files": devFiles, "labels": devLabels}
        testData = {"files": testFiles, "labels": testLabels}

        dataDict = {"train": trainData, "val": valData, "dev": devData, "test": testData}

        return dataDict


    def getLabel(self, fileName: str) -> int:
        """
        get the file name and return the label of the audio

        Parameters:
        -------------------------
        fileName: name of the audio file (full path)

        Return
        -------------------------
        label: label of the file name (int)
        
        """

        category = fileName.split("/")[-2]

        label = categories.get(category, categories["_background_noise_"])

        return label

@dataclass
class PreProcess():

    """
    Preprocess the audio files into MFCC features

    Instances
    ---------------
    dataset: an instance of dataset class
    sr: int, sampling rate
    nMfcc: int, the number of mfcc
    mfccLength: int, the length of the mfcc
    """

    dataset: Dataset
    sr: int
    nMfcc: int
    mfccLength: int

    def audiotoMfccFile(self, path:str) -> np.ndarray:
        
        """
        helper function to load audio file using librosa library. Applied MFCC and return MFCC features.

        Parameters
        ---------------------
        path: str, path of the audio file
        sr: int, sampling rate 
        nMfcc: int, Number of MFCCs to return
        mfccLength: int, Length of MFCC features for each audio input

        Return
        ---------------------
        mfccFeatures: np.ndarray, the mfcc features
        
        
        """
        audio, _ = librosa.load(path, sr = self.sr)

        mfccFeatures = librosa.feature.mfcc(y = audio,
                                             n_mfcc = self.nMfcc,
                                             sr = self.sr)
        
        if (self.mfccLength > mfccFeatures.shape[1]):
            padding_width = self.mfccLength - mfccFeatures.shape[1]
            mfccFeatures = np.pad(mfccFeatures, 
                                pad_width =((0, 0), 
                                            (0, padding_width)),
                                            mode ='constant')
        else:
            mfccFeatures = mfccFeatures[:, :self.mfccLength]

        return mfccFeatures

    def audio_files_to_numpy(self, fileType: str) -> \
        (np.ndarray, np.ndarray):

        """
        return numpy file of all the audio files

        Parameters
        -----------------
        fileType: list, [train, val, test, dev]

        return
        -------------------
        mfccFeatures: np.ndarray, a numpy matrix containing all the audio files
        labels: np.ndarray, a numpy array containing all the labels
        """
        
        files, labels = self.loadDatasetDict(datasetDict=self.dataset.getDataDict(), fileType=fileType)
        
        mfccFeatures = self.npyFiles(files = files, sr = self.sr,
                          nMfcc= self.nMfcc, mfccLength= self.mfccLength)
        
        return mfccFeatures, labels

    def save_numpy (self, path: str, fileType: str, mfccFeatures: np.ndarray,
                     labels: np.ndarray) -> None:
        """
        Save the MFCC features and labels in .npy format

        Parameters
        --------------
        path: str, path for saving
        fileType: list, [train, val, dev, test]
        mfccFeatures: np.ndarray, all MFCC features in 3d array format
        labels: np.ndarray, all labels in array format

        Return
        ---------------
        None
        """
        np.save (path + '/' + fileType +'.npy', mfccFeatures)
        np.save (path + '/' + fileType +"Labels.npy", labels)
           
    def loadDatasetDict(self, datasetDict: dict, fileType :str) -> (list, list):
        
        """
        helper function for dumpAudioFiles for loading files and labels of from dataset dictionary

        Parameters
        ---------------
        datasetDict: dict, a dictionary of paths and labels. Details: Dataset.getDataDict
        fileType: str, train, val, dev or test

        Return
        ---------------
        files: list, a list of all file paths
        labels: list, a list of all labels
        """

        innerDict: dict = datasetDict.get(fileType)
        files : list = innerDict.get('files')
        labels : list = innerDict.get('labels')

        return files, labels


    def npyFiles(self,files: list, sr: int, nMfcc: int, mfccLength: int)\
        -> list:

        """
        helper function for audioFilesNumpy for changing list of files into .npy file

        Parameters
        ---------------
        files: list, a list containing file paths to audios
        sr: int, sampling rate 
        nMfcc: int, Number of MFCCs to return
        mfccLength: int, Length of MFCC features for each audio input

        Return
        ---------------
        mfccFeaturesAll: list, the list of all MFCC features
        """

        mfccFeaturesAll = []

        for file in tqdm(files):
            mfccFeatures = self.audiotoMfccFile(path = file,
                                                sr = sr,
                                                nMfcc = nMfcc,
                                                mfccLength= mfccLength)
            mfccFeaturesAll.append(mfccFeatures)
        mfccFeaturesAll = np.array(mfccFeaturesAll)

        return mfccFeaturesAll


