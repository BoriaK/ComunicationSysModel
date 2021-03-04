import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

# environment params:
G_l = 1  # product of Tx and Rx antenna field radiation patterns in the LOS direction
G_r = 1  # product of Tx and Rx antenna field radiation patterns in the reflection direction
ht = 1  # distance of transmitter from the ground - [m]
hr = 2  # distance of receiver from the ground - [m]
dist = 10  # the ground distance from Tx antenna till Rx antenna - [m]
# model params:
Theta = np.arctan((ht + hr) / dist)  # the angle of the (hitting and) reflecting wave from the ground [rad]
x1 = ht / np.sin(Theta)  # distance from Tx antenna till impact point - [m]
x2 = hr / np.sin(Theta)  # distance from impact point till Rx antenna - [m]
los = np.sqrt((ht - hr) ** 2 + dist ** 2)  # LOS distance from Tx antenna till Rx antenna - [m] ?
# Tao = (x1 + x2 - los) / 3e8  # time delay of the reflected signal
Tao = 5e-3  # time delay of the reflected signal
# additional params:
R = -1  # ground reflection coefficient

# Signal Params
freq = 10  # signal frequency - [Hz]
t = np.arange(0, 1, 0.001)  # time vector corresponding to the input signal
Fc = 0  # Carrier frequency of the entire signal - [Hz]
# Tao = np.pi / 3
Sig = np.sin(2 * np.pi * freq * t)
Sig2 = np.sin(2 * np.pi * freq * (t - Tao))

Lambda = freq / 3e8  # wavelength of the transmitted signal - [m]

outSignal = np.real(((Lambda / (4 * np.pi)) * ((np.sqrt(G_l) * Sig * np.exp(-1j * 2 * np.pi / Lambda)) / los) +
                     (R * (np.sqrt(G_r)) * Sig2 * np.exp(-1j * 2 * np.pi * (x1 + x2) / Lambda)) / (
                             x1 + x2)) * np.exp(1j * 2 * np.pi * Fc * t))

plt.figure()
plt.plot(t, Sig, t, Sig2, t, outSignal)
plt.xlabel('Time')
plt.ylabel('S(t)')
plt.title('2ray Model - sin(t)')
plt.grid()
plt.legend(['s(t)', 's(t-Tau), outSignal'])
plt.show()

# print()