import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math


def OFDM_FFT_Tx(input_data):
    Num_Dta_chnk = int(len(input_data) / 56)
    GI = 0.8 * 1e-6  # 0.8[uS] Long GI
    Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
    Delta_F = 1 / Tsym  # channel spacing
    F = 1 / (Tsym + GI)  # analog Frequency
    F_samp = 20 * 1e6  # sample frquency 20MHz

    t_sym = np.arange(0, Tsym, 1 / F_samp)
    F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t_sym))

    S_t_w_CP = np.zeros((len(t_sym) + 16) * Num_Dta_chnk, dtype=np.complex)
    # S_f = np.zeros((Num_Dta_chnk, len(F_axis)), dtype=np.complex)
    for chnk in range(Num_Dta_chnk):
        S_f_chnk = np.zeros(len(F_axis), dtype=np.complex)
        Dta_Tx_chnk = input_data[range(chnk * 56, (chnk + 1) * 56)]
        # prepare the data in frequency domain:
        skip = 0
        for k in range(len(F_axis)):
            if (k - len(F_axis) / 2) < -28 or (k - len(F_axis) / 2) == 0 or (k - len(F_axis) / 2) > 28:
                skip += 1
                S_f_chnk[k] = 0
            else:
                S_f_chnk[k] = Dta_Tx_chnk[k - skip]

        # S_f[chnk] = S_f_chnk

        # single OFDM symbol in Frequency domain
        # plt.figure()
        # plt.plot(F_axis, np.abs(S_f_chnk))
        # plt.xlabel('Frequency')
        # plt.ylabel('S(f)')
        # plt.grid()
        # plt.title('OFDM symbol in frequency domain')
        # plt.show()

        # prepare time domain signal using IFFT:
        S_t_chnk = np.fft.ifft(np.fft.fftshift(S_f_chnk), n=64)

        # single OFDM symbol in Time domain
        # plt.figure()
        # plt.plot(t, S_t_chnk)
        # plt.xlabel('Time')
        # plt.ylabel('S(t)')
        # plt.grid()
        # plt.title('OFDM symbol in time domain')
        # plt.show()

        # Parallel to serial: ?

        # inserting cyclic prefix:
        S_t_chnk_w_CP = np.zeros(int(len(S_t_chnk) + 16), dtype=np.complex)
        CP = S_t_chnk[range(int(len(S_t_chnk) - 16), len(S_t_chnk))]
        S_t_chnk_w_CP[range(16)] = CP
        S_t_chnk_w_CP[range(16, len(S_t_chnk_w_CP))] = S_t_chnk

        t_sym_w_CP = np.arange(0, Tsym + GI, 1 / F_samp)  # new Symbol time includes cyclic prefix

        # plt.figure()
        # plt.plot(t_sym_w_CP, S_t_chnk_w_CP)
        # plt.xlabel('Time')
        # plt.ylabel('S(t) with GI')
        # plt.title('single OFDM symbol with CP in time domain')
        # plt.grid()
        # plt.show()

        S_t_w_CP[chnk * len(S_t_chnk_w_CP):(chnk + 1) * len(S_t_chnk_w_CP)] = S_t_chnk_w_CP

    t_w_CP = np.arange(0, (Tsym + GI) * Num_Dta_chnk, 1 / F_samp)
    # plt.figure()
    # plt.plot(t_w_CP, S_t_w_CP)
    # plt.xlabel('Time')
    # plt.ylabel('S(t) with GI')
    # plt.title('OFDM signal with CP in time domain')
    # plt.grid()
    # plt.show()

    # D/A converter:
    # up sample by factor of 8
    # upsample using ZOH interpolation
    up = 8
    spaced_S_t_w_CP_up = np.zeros(up * len(S_t_w_CP), dtype=np.complex)
    spaced_S_t_w_CP_up[::up] = S_t_w_CP
    g_ZOH = np.ones(up, dtype=np.complex)
    S_t_w_CP_up = signal.lfilter(g_ZOH, 1, spaced_S_t_w_CP_up)
    t_w_CP_up = np.arange(0, (Tsym + GI) * Num_Dta_chnk, 1 / (up * F_samp))

    # plt.figure()
    # plt.plot(t_w_CP[:int(len(t_w_CP)/Num_Dta_chnk)], S_t_w_CP[:int(len(S_t_w_CP)/Num_Dta_chnk)],
    #          t_w_CP_up[:int(len(t_w_CP_up)/Num_Dta_chnk)], S_t_w_CP_up[:int(len(S_t_w_CP_up)/Num_Dta_chnk)])
    # plt.xlabel('Time')
    # plt.ylabel('S(t) with GI')
    # plt.title('regular vs upsampled OFDM symbol with CP in time domain')
    # plt.grid()
    # plt.legend(['s(t)', 'upsampled s(t)'])
    # plt.show()

    return S_t_w_CP_up, up

# def main():
#     # OFDM_Oscilator_Tx()
#     OFDM_FFT_Tx(Dta_Tx)


# main()
