#importing libraries

from pylab import *
import scipy.io.wavfile as wavfile  #for reading and writing .wav files
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



# Frequencies w0 and w1
w0 = 1000
w1 = 3000
w_s= 8192 #sampling freq

# data points
N = np.linspace(0,511,512)

# generating sinusoids
x1 = np.sin(w0*N/w_s) # sinusoid 1
x2 = np.sin(w1*N/w_s) # sinusoid 2

#convolving the two signals

y = np.convolve(x1,x2,'full')

#computing DFT of convolved output

Y = np.fft.fft(y)
f1 = np.linspace(0,511,len(Y)) #freq

plt.plot(f1,np.abs(Y))
plt.title('DFT Spectrum of convolved output')
plt.xlabel('frequency')
plt.ylabel('|Y|')
plt.grid()
plt.show()


#now from the other side

X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)

Y_alt = X1*X2
f2 = np.linspace(0,511,len(Y_alt)) #freq 
plt.plot(f2,np.abs(Y_alt))
plt.title('Spectrum of the product of the DFTs')
plt.xlabel('frequency')
plt.ylabel('|Y|')
plt.grid()
plt.show()

plt.plot(X1)
plt.show()

# Audio in is the audio file we're using to compute the DFT

Audio_in = "TestHello1.wav"  #input audio file
fs,data  = wavfile.read(Audio_in)  #fs = initial sampling freq of the source,data = corresponding numpy array
N_sample = len(data) #No of samples initially

#computing DFT
Data = np.fft.fft(data)

plt.plot(f)
plt.show()
