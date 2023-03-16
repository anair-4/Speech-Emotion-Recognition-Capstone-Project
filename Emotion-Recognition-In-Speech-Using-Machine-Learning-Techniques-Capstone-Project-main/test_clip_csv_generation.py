import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from sklearn.utils import shuffle
import os
import pandas as pd
import glob 
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys
import pitch
import scipy.io.wavfile as wav
#import pyAudioAnalysis
from python_speech_features import logfbank

import warnings

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

mypath = './Test_Clips'
list_of_files = []
label_list = []
pitch_list = []
lfbe_list = []
name_list = []
path_dict = {}

for r, d, f in os.walk(mypath):
    for file in f:
        if file.endswith(".wav"):
            list_of_files.append(file)
            file_path = os.path.join(r, file)
            path_dict[file] = file_path
              
               

for file in list_of_files:
    if file[0] == 'A':
        label_list.append('anger')
    elif file[0] == 'F':
        label_list.append('fear')
    elif file[0] == 'H':
        label_list.append('happiness')
    elif file[0] == 'N':
        label_list.append('neutral')
    elif file[0] == 'S' and file[2]=='D':
        label_list.append('sadness')
    elif file[0] == 'S' and file[2] == 'R':
        label_list.append('sarcasm')
    else:
        print(file)
    
    p = pitch.find_pitch('./Test_Clips/'+file)
    pitch_list.append(p)

    rate,sig = wav.read('./Test_Clips/'+file)
    lfbank_e = logfbank(sig, rate)
    lfb_e = np.mean(lfbank_e)
    lfbe_list.append(lfb_e)
    name_list.append(file)

    print(file)

labels = pd.DataFrame(label_list)
pitches = pd.DataFrame(pitch_list)
lfbe_column = pd.DataFrame(lfbe_list)
name_column = pd.DataFrame(name_list)

print(labels)

df = pd.DataFrame(columns=['feature'])
bookmark=0
for file in path_dict:
    X, sample_rate = librosa.load(path_dict[file], res_type='kaiser_fast', duration = 2.5, sr=22050*2,offset=1)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    df.loc[bookmark] = [feature]
    bookmark=bookmark+1
print("ok")


# In[28]:


df3 = pd.DataFrame(df['feature'].values.tolist())
newdf_2 = pd.concat([df3,pitches], axis=1)
newdf_1 = pd.concat([newdf_2,lfbe_column], axis=1)
newdf_0 = pd.concat([newdf_1, labels], axis =1)
newdf = pd.concat([newdf_0, name_column], axis=1)
print("labels")
renamed_newdf = newdf.rename(index=str, columns={"0": "label"})
renamed_newdf = shuffle(newdf)
renamed_newdf=renamed_newdf.fillna(0)
renamed_newdf.to_csv('Test_Clips_1.csv')
print("ok")