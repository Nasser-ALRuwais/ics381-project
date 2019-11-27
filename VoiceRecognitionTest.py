import os
import librosa
import sounddevice as sd
import soundfile as sf
import IPython.display as ipd
import numpy as np
from flask import Flask
from keras.models import load_model
model=load_model('ics381-model.hdf5')

labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(labels)
classes= list(le.classes_)

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]


samplerate = 16000  
duration = 1 # seconds
filename = 'yes.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)

samples, sample_rate = librosa.load('yes.wav', sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples,rate=8000)  

print(predict(samples))