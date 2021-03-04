import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

Lambda = 1  # wavelength of the transmitted signal
G_l = 1  # product of Tx and Rx antenna field radiation patterns in the LOS direction
G_r = 1  # product of Tx and Rx antenna field radiation patterns in the reflection direction
ht = 1  # distance of transmitter from the ground - [m]
hr = 2  # distance of receiver from the ground - [m]
dist = 10  # the ground distance from Tx antenna till Rx antenna - [m]
Theta = np.arctan((ht+hr)/dist)  # the angle of the (hitting and) reflecting wave from the ground [rad]
x1 = ht/np.sin(Theta)  # distance from Tx antenna till impact point - [m]
x2 = hr/np.sin(Theta)  # distance from impact point till Rx antenna - [m]
los = np.sqrt((ht-hr)**2+dist**2)  # LOS distance from Tx antenna till Rx antenna - [m] ?
R = -1  # ground reflection coefficient
Fc = 1  # Carrier frequency of the entire signal
Time_Vec = 1  # time vector corresponding to the input signal


def TwoRayModel(inSignal):
    outSignal = np.real(((Lambda / (4 * np.pi)) * ((np.sqrt(G_l) * inSignal * np.exp(-1j * 2 * np.pi / Lambda)) / los) +
                         (R * (np.sqrt(G_r)) * inSignal * np.exp(-1j * 2 * np.pi * (x1 + x2) / Lambda)) / (
                                     x1 + x2)) * np.exp(1j * 2 * np.pi * Fc * Time_Vec))
    return outSignal
