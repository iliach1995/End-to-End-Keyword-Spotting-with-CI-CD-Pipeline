
from dataclasses import dataclass
import numpy as np
from parameters import *
import librosa

@dataclass
class AudioProcess():

    sr: int 
    nMfcc: int
    mfccLength: int

    def audiotoMfccFile(self, path:str) -> np.ndarray:
        
        """
        load audio file using librosa library. Applied MFCC and return MFCC features.

        Parameters
        ---------------------
        path: str, path of the audio file

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
    
if __name__ == '__main__':
    audioProcess = AudioProcess(SAMPLE_RATE, N_MFCC,MFCC_LENGTH)