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

frequencia_amostragem, data = wavfile.read('dog.wav')
length = data.shape[0] / frequencia_amostragem
time = np.linspace(0. , length, data.shape[0])


N = 1024
FFT = abs(fftpack.fft(data))
freqs = fftpack.fftfreq(N, (1.0/frequencia_amostragem))


plt.plot(freqs[range(N//2)], FFT[range(N//2)])
plt.title("Espectro fft")
plt.xlabel('Frequencia (Hz)')
plt.ylabel('Amplitude')
plt.show()

