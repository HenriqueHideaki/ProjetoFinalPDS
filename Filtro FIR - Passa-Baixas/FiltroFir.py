import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy import fftpack
from scipy.io import wavfile

frequencia_amostragem, data = wavfile.read('dog.wav')
length = data.shape[0] / frequencia_amostragem
time = np.linspace(0. , length, data.shape[0])

#Filtro com os coeficientes gerados pelo PyFda
hamming = np.genfromtxt("firPyFda.csv", delimiter=",")


