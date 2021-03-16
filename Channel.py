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
dist = 300  # the ground distance between Tx antenna and Rx antenna - [m]


def TwoRayModel(inSignal, up):
    # Signal Params:
    GI = 0.8 * 1e-6  # 0.8[uS] Long GI
    Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
    F_samp = 20 * 1e6  # sample frquency 20MHz
    Num_Dta_chnk = int((len(inSignal) / up) * 0.8 / 64)
    freq = 312.5e3  # signal frequency - [Hz]; each spectral component should have different freq?
    t = np.arange(0, (Tsym + GI) * Num_Dta_chnk,
                  1 / (up * F_samp))  # time vector corresponding to the input signal [Sec]
    Fc = 0  # Carrier frequency of the entire signal - [Hz]; Fc = 0 -> BB

    # model params:
    Theta = np.arctan((ht + hr) / dist)  # the angle of the (hitting and) reflecting wave from the ground [rad]
    x1 = ht / np.sin(Theta)  # distance from Tx antenna till impact point - [m]
    x2 = hr / np.sin(Theta)  # distance from impact point till Rx antenna - [m]
    los = np.sqrt((ht - hr) ** 2 + dist ** 2)  # LOS distance from Tx antenna till Rx antenna - [m] ?
    Tao = (x1 + x2 - los) / 3e8  # time delay of the reflected signal [sec]
    TaoSamp = int(np.around(Tao * F_samp))  # number of samples equal to Tau delay [sec]
    # TaoSamp = 1

    # additional params:
    R = -1  # ground reflection coefficient

    Lambda = 3e8 / freq  # wavelength of the transmitted signal - [m]
    inSignal2 = np.zeros(len(inSignal), dtype=np.complex)
    inSignal2[TaoSamp:] = inSignal[TaoSamp:]
    outSignal = np.real((Lambda / (4 * np.pi)) * (
            ((np.sqrt(G_l) * inSignal * np.exp((-1j * 2 * np.pi * los) / Lambda)) / los) + (
            (R * (np.sqrt(G_r)) * inSignal2 * np.exp((-1j * 2 * np.pi * (x1 + x2)) / Lambda)) / (
            x1 + x2))) * np.exp(1j * 2 * np.pi * Fc * t))

    plt.figure()
    plt.plot(t[:int(len(t)/Num_Dta_chnk)], inSignal[:int(len(inSignal)/Num_Dta_chnk)], t[:int(len(t)/Num_Dta_chnk)], inSignal2[:int(len(inSignal2)/Num_Dta_chnk)], t[:int(len(t)/Num_Dta_chnk)], outSignal[:int(len(outSignal)/Num_Dta_chnk)])
    # plt.plot(t, outSignal)
    plt.xlabel('Time')
    plt.ylabel('S(t)')
    plt.title('2ray Model')
    plt.grid()
    plt.legend(['s(t)', 's(t-Tau)', 'outSignal'])
    plt.show()

    return outSignal
