import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
import sunau;
import librosa
import librosa.display
print("hello");
filename="bird.wav"; 
y, sr = librosa.core.load(filename);
print y;
 #mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, n_mfcc=13);
print(sr);
a=y;
b=[(ele/2**8.)*2-1 for ele in a] 
#librosa.feature.spectral_centroid();
#c=librosa.stft(y=y);
print("len",len(c));
print(c);

#c = fft(data) # calculate fourier transform (complex numbers list)
d = len(b)/1  # you only need half of the fft list (real signal symmetry)
#librosa.display.specshow(abs(c[:(d-1)]), x_axis='time')
#plt.plot(abs(c[:(d-1)]),'r') 
#plt.show()
