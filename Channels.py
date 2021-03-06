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
dist = 30  # the ground distance between Tx antenna and Rx antenna - [m]


def NoChannel(inSignal):
    outSignal = inSignal

    return outSignal


def TwoRayModel(inSignal, up):
    States = {1: 'Path Loss + Phase Shift + MP', 2: 'Path Loss + MP', 3: 'Phase Shift + MP', 4: 'Relative Path Loss, '
                                                                                                'Phase Shift + MP',
              5: 'Relative Path Loss and Phase Shift + MP'}
    State = States[4]

    # Signal Params:
    GI = 0.8 * 1e-6  # 0.8[uS] Long GI
    Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
    F_samp = 20 * 1e6  # sample frequency 20MHz
    Num_Dta_chnk = int((len(inSignal) / up) * 0.8 / 64)
    t = np.arange(0, (Tsym + GI) * Num_Dta_chnk,
                  1 / (up * F_samp))  # time vector corresponding to the input signal [Sec]
    Fc = 2.4e9  # Carrier frequency of the entire signal - [Hz]; Fc = 0 -> BB
    freq = Fc  # signal frequency - [Hz]; choose max carrier frequency for the whole signal; not sure
    # whether choose Fc?

    # model params:
    Theta = np.arctan((ht + hr) / dist)  # the angle of the (hitting and) reflecting wave from the ground [rad]
    x1 = ht / np.sin(Theta)  # distance from Tx antenna till impact point - [m]
    x2 = hr / np.sin(Theta)  # distance from impact point till Rx antenna - [m]
    los = np.sqrt((ht - hr) ** 2 + dist ** 2)  # LOS distance from Tx antenna till Rx antenna - [m] ?
    # los = 12.5  # for debug
    Tao = (x1 + x2 - los) / 3e8  # time delay of the reflected signal [sec]
    # Tao = 1.03 * 0.8e-6  # for testing
    TaoSamp = int(np.around(Tao * F_samp))  # number of samples equal to Tau delay [sec]
    # TaoSamp = 1

    # additional params:
    Er = 4.5  # the relative permittivity of concrete
    Xh = np.sqrt(Er - math.pow(np.cos(Theta), 2))  # Horizontal Polarization
    Xv = Xh / Er  # Vertical Polarization
    # X = Xv  # decide the wave is vertically polarized
    X = Xh  # decide the wave is horizontally polarized
    # R = -1
    R = (np.sin(Theta) - X) / (np.sin(Theta) + X)  # ground reflection coefficient

    Lambda = 3e8 / freq  # wavelength of the transmitted signal - [m]
    inSignal2 = np.zeros(len(inSignal), dtype=np.complex)
    inSignal2[TaoSamp:] = inSignal[TaoSamp:]

    # Original Formula:
    # LOS_Component = (Lambda / (4 * np.pi)) * ((np.sqrt(G_l) * inSignal * np.exp((-1j * 2 * np.pi * los) / Lambda)) /
    #                                           los)
    # SecondRay = (Lambda / (4 * np.pi)) * ((R * (np.sqrt(G_r)) * inSignal2 * np.exp((-1j * 2 * np.pi *
    #                                                                                 (x1 + x2)) / Lambda)) / (
    #                                                   x1 + x2))
    # outSignal = np.real((LOS_Component + SecondRay) * np.exp(1j * 2 * np.pi * Fc * t))
    # Complex Base Band Signal:
    if State == 'Path Loss + Phase Shift + MP':
        # Gain (Path Loss) + Phase Shift + Multi Path:
        LOS_Component = (Lambda / (4 * np.pi)) * (
                (np.sqrt(G_l) * inSignal * np.exp((-1j * 2 * np.pi * los) / Lambda)) /
                los)
        SecondRay = (Lambda / (4 * np.pi)) * (
                (R * (np.sqrt(G_r)) * inSignal2 * np.exp((-1j * 2 * np.pi * (x1 + x2)) /
                                                         Lambda)) / (x1 + x2))

    elif State == 'Path Loss + MP':
        # Gain (Path Loss) + Multi Path Only:
        LOS_Component = (Lambda / (4 * np.pi)) * ((np.sqrt(G_l) * inSignal) / los)
        SecondRay = (Lambda / (4 * np.pi)) * ((R * (np.sqrt(G_r)) * inSignal2) / (x1 + x2))

    elif State == 'Phase Shift + MP':
        # Phase Shift + Multi Path Only:
        LOS_Component = np.sqrt(G_l) * inSignal * np.exp((-1j * 2 * np.pi * los) / Lambda)
        SecondRay = R * np.sqrt(G_r) * inSignal2 * np.exp((-1j * 2 * np.pi * (x1 + x2)) / Lambda)
    elif State == 'Relative Path Loss, Phase Shift + MP':
        # Phase Shift + Multi Path Only:
        LOS_Component = np.sqrt(G_l) * inSignal * np.exp((-1j * 2 * np.pi * los) / Lambda)
        # LOS_Component = np.sqrt(G_l) * inSignal * np.exp(-1j * 0.25 * np.pi)
        # LOS_Component = np.sqrt(G_l) * inSignal
        # SecondRay = R * np.sqrt(G_r) * inSignal2 * np.exp((-1j * 2 * np.pi * (x1 + x2)) / Lambda)
        K = 0.9  # K is a weight we give second component
        SecondRay = K * R * np.sqrt(G_r) * inSignal2 * np.exp((-1j * 2 * np.pi * (x1 + x2)) / Lambda)
        # SecondRay = (R * np.sqrt(G_r) * inSignal2)

    elif State == 'Relative Path Loss and Phase Shift + MP':
        # Multi Path Only:
        # Normalize the path loss and the phase shift to the reflecting component:
        LOS_Component = np.sqrt(G_l) * inSignal
        SecondRay = (R * np.sqrt(G_r) * inSignal2 * np.exp((-1j * 2 * np.pi * (x1 + x2 - los)) / Lambda)) * (
                los / (x1 + x2))


    outSignal = LOS_Component + SecondRay
    # ######### for debug:####################################################################
    Gain = np.abs(R) * np.sqrt(G_r) * (los / (x1 + x2)) * K
    # LOSPhase = (-1 * 2 * los) / Lambda  # Total Phase LOS component[pi-rad]
    LOSPhase = -np.mod((2 * los/Lambda), 2)
    # SecondPhase = (-1 * 2 * (x1 + x2)) / Lambda + 1  # Total Phase second component[pi-rad]
    SecondPhase = -np.mod(2 * (x1 + x2) / Lambda - 1, 2)
    RelativePhase = SecondPhase - LOSPhase  # Relative Phase [pi-rad]
    print('Gain = ', Gain)
    print('LOS Phase = ', LOSPhase)
    print('Second Phase = ', SecondPhase)
    print('Relative Phase = ', RelativePhase)
    fixedSignal = (LOS_Component + SecondRay) * np.exp(-1j*LOSPhase*np.pi)
###########################################################################################################
    # Recieved Power if Pt = 1:
    Pr = math.pow(
        np.abs(np.sqrt(G_l) + K * R * np.sqrt(G_r) * np.exp((-1j * 2 * np.pi * (x1 + x2 - los)) / Lambda) * (
                los / (x1 + x2))), 2)
    print('Pr/Pt = ', Pr)

    # plt.figure()
    # plt.plot(t[:int(len(t) / Num_Dta_chnk)], LOS_Component[:int(len(inSignal) / Num_Dta_chnk)],
    #          t[:int(len(t) / Num_Dta_chnk)],
    #          SecondRay[:int(len(inSignal2) / Num_Dta_chnk)], t[:int(len(t) / Num_Dta_chnk)],
    #          outSignal[:int(len(outSignal) / Num_Dta_chnk)])
    # plt.xlabel('Time')
    # plt.ylabel('S(t)')
    # plt.title('2ray Model')
    # plt.grid()
    # plt.legend(['s(t)', '\u03B1*s(t-Tau)', 'outSignal'])
    # plt.show()

    # return outSignal
    return fixedSignal
