import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

# environment params:
G_l = 1  # product of Tx and Rx antenna field radiation patterns in the LOS direction
G_r = 1  # product of Tx and Rx antenna field radiation patterns in the reflection direction
ht = 1  # distance of the transmitter from the ground - [m]
hr = 2  # distance of the receiver from the ground - [m]
dist = 20  # the ground distance between Tx antenna and Rx antenna - [m]
# model params:
Theta = np.arctan((ht + hr) / dist)  # the angle of the (hitting and) reflecting wave from the ground [rad]
x1 = ht / np.sin(Theta)  # distance from Tx antenna till impact point - [m]
x2 = hr / np.sin(Theta)  # distance from impact point till Rx antenna - [m]
los = np.sqrt((ht - hr) ** 2 + dist ** 2)  # LOS distance from Tx antenna till Rx antenna - [m] ?
Tao = (x1 + x2 - los) / 3e8  # time delay of the reflected signal
# Tao = 1.03 * 0.8e-6  # time delay of the reflected signal
# additional params:
Er = 4.5  # the relative permittivity of concrete
Xh = np.sqrt(Er - math.pow(np.cos(Theta), 2))  # Horizontal Polarization
Xv = Xh / Er  # Vertical Polarization
X = Xh  # decide the wave is horizontally polarized
# R = -1
R = (np.sin(Theta) - X) / (np.sin(Theta) + X)  # ground reflection coefficient

# Signal Params
freq = 312.5e3  # signal frequency - [Hz]
t = np.arange(0, 10e-6, 1e-7)  # time vector corresponding to the input signal [Sec]
Fc = 0  # Carrier frequency of the entire signal - [Hz]
Sig = np.sin(2 * np.pi * freq * t)
Sig2 = np.sin(2 * np.pi * freq * (t - Tao))

Lambda = 3e8 / freq  # wavelength of the transmitted signal - [m]
# Original Formula:
# LOS_Component = (Lambda / (4 * np.pi)) * ((np.sqrt(G_l) * Sig * np.exp((-1j * 2 * np.pi * los) / Lambda)) / los)
# SecondRay = (Lambda / (4 * np.pi)) * ((R * (np.sqrt(G_r)) * Sig2 * np.exp((-1j * 2 * np.pi * (x1 + x2)) / Lambda)) / (
#             x1 + x2))
# outSignal = np.real((LOS_Component + SecondRay) * np.exp(1j * 2 * np.pi * Fc * t))

# Multi Path Only:
# take the complex value
# Base Band signal
# Normalize the path loss and the phase shift to the reflecting component
LOS_Component = np.sqrt(G_l) * Sig
SecondRay = R * np.sqrt(G_r) * Sig2 * np.exp((-1j * 2 * np.pi * (x1 + x2 - los)) / Lambda) * (los / (x1 + x2))
outSignal = LOS_Component + SecondRay

plt.figure()
plt.plot(t, LOS_Component, t, SecondRay, t, outSignal)
# plt.plot(t, outSignal)
plt.xlabel('Time')
plt.ylabel('S(t)')
plt.title('2ray Model - sin(t)')
plt.grid()
plt.legend(['s(t)', '\u03B1*s(t-Tau)', 'outSignal'])
plt.show()

# print()
