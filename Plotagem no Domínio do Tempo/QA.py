import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf 
import IPython.display as ipd
from scipy import fftpack
import scipy.fftpack as fftpk

from scipy.fft import fft, fftfreq
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy import signal

#samplerate -> frequencia de amostragem e o dado


#Construindo o sinal no dominio do tempo 

#importando o audio 


# data.shape[0] -> numero de amostras
#samplerate -> frequencia de amostragem -> fs 

#QUESTÃO A


#Length -> Tempo total = numero de amostras / frequencia de amostragem
frequencia_amostragem, data = wavfile.read('dog.wav')
length = data.shape[0] / frequencia_amostragem
time = np.linspace(0. , length, data.shape[0])

plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

print('\n\n')
print(f'Tempo de duracao do audio: {length} s')
print(f'Numero de amostras :       {data.shape[0]} ')
print(f'Frequencia de amostragem:   {frequencia_amostragem}Hz')









