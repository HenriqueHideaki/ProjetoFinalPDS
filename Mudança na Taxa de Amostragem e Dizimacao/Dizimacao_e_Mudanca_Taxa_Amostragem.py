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



hamming = np.genfromtxt("firPyFda.csv", delimiter=",")


def overlapAndAdd(signal, fir):
    max_len = max(len(signal), len(fir))

    x = np.zeros(max_len)
    h = np.zeros(max_len)
    x[:len(signal)] = signal
    h[:len(fir)] = fir

    X = np.fft.fft(x)
    H = np.fft.fft(h)
    Y = X * H

    return np.real(np.fft.ifft(Y))


conv = overlapAndAdd(data, hamming)


plt.plot(time, data, 'b-', label='Original')
plt.plot(time, conv, 'y-', label='Filtro')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

f, Pxx_spec = welch(data, frequencia_amostragem, 'flattop', 512, scaling='spectrum')
f_filtered, Pxx_spec_filtered = welch(
    conv, frequencia_amostragem, 'flattop', 512, scaling='spectrum')

plt.semilogy(f, Pxx_spec, 'b-', label='Original')
plt.semilogy(f_filtered, Pxx_spec_filtered, 'y-', label='Filtrado')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()


M = 2
x = conv[0:-1:M]  

time2 = np.linspace(0., length, x.shape[0])

plt.plot(time, data, 'b-', label='Original')
plt.plot(time, conv, 'y-', label='Filtrado')
plt.plot(time2, x, 'r-', label='Filtrado e dizimado')
plt.title(f'M = {M}')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

f, Pxx_spec = welch(data, frequencia_amostragem, 'flattop', 512, scaling='spectrum')
f_filtered, Pxx_spec_filtered = welch(
    conv, frequencia_amostragem, 'flattop', 512, scaling='spectrum')
f_decimated, Pxx_spec_decimated = welch(
    x, frequencia_amostragem, 'flattop', 512, scaling='spectrum')

# Plota o espectro do sinal para frequencias normalizadas entre 0 1 pi
plt.semilogy(f, Pxx_spec, 'b-', label='Original')
plt.semilogy(f_filtered, Pxx_spec_filtered, 'y-', label='Filtrado')
plt.semilogy(f_decimated, Pxx_spec_decimated,
             'r-', label='Filtrado e dizimado')
plt.title(f'M = {M}')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()
