import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math


Num_Dta_chnk = int(10 * 1e3)  # number of data chunks
# random 56 symbols of data per packet
rng = np.random.default_rng()

# 16QAM
M = 16
# infase data
m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56*Num_Dta_chnk, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))
# quadrature data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56*Num_Dta_chnk, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))

Dta_Tx = m_i + 1j * m_q


def OFDM_Oscilator_Tx():
    GI = 0.8 * 1e-6  # 0.8[uS] Long GI
    Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
    Delta_F = 1 / Tsym
    F = 1 / (Tsym + GI)  # analog Frequency
    F_samp = 20 * 1e6  # sample frquency 20MHz

    t_sym = np.arange(0, Tsym, 1 / F_samp)
    F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t_sym))

    CW_tk = np.zeros((len(F_axis), len(t)), dtype=np.complex)
    S_tk = np.zeros((len(F_axis), len(t)), dtype=np.complex)

    # prepare the data in frequency domain:
    skip = 0
    for k in range(len(F_axis)):
        if (k - len(F_axis) / 2) < -28 or (k - len(F_axis) / 2) == 0 or (k - len(F_axis) / 2) > 28:
            skip += 1
        else:
            # CW_tk[k][:] = (1 / np.sqrt(Tsym)) * np.exp(1j * 2 * np.pi * (k - len(F_axis) / 2) * Delta_F * t)
            CW_tk[k][:] = np.exp(1j * 2 * np.pi * (k - len(F_axis) / 2) * Delta_F * t)
        S_tk[k][:] = Dta[k - skip] * CW_tk[k]

    S_t = np.sum(S_tk, axis=0)

    # Plot the OFDM Symbol in Time domain
    # plt.figure()
    # plt.plot(t, S_t)
    # plt.xlabel('Time')
    # plt.ylabel('S_(t)')
    # plt.grid()
    # plt.title('OFDM symbol in Time domain')
    # plt.show()

    ############################ Plot some Carriers of the OFDM symbol in time#############################
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(t, S_tk[4])
    # plt.xlabel('Time')
    # plt.ylabel('S_4(t)')
    # plt.grid()

    # plt.subplot(3, 1, 2)
    # plt.plot(t, S_tk[15])
    # plt.xlabel('Time')
    # plt.ylabel('S_15(t)')
    # plt.grid()

    # plt.subplot(3, 1, 3)
    # plt.plot(t, S_tk[25])
    # plt.xlabel('Time')
    # plt.ylabel('S_25(t)')
    # plt.grid()
    # plt.suptitle('some Carriers of the OFDM symbol in time')
    # plt.show()
    ######################################################################################################3

    # OFDM Symbol in Frequency domain:
    S_f = np.fft.fft(S_t)

    # plt.figure()
    # plt.plot(F_axis, np.abs(np.fft.fftshift(S_f)))
    # plt.xlabel('Frequency')
    # plt.ylabel('S(f)')
    # plt.title('OFDM symbol in frequency domain')
    # plt.grid()
    # plt.show()

    # inserting cyclic prefix:
    S_t_w_CP = np.zeros(int(len(S_t) + 16), dtype=np.complex)
    CP = S_t[range(int(len(S_t) - 16), len(S_t))]
    S_t_w_CP[range(16)] = CP
    S_t_w_CP[range(16, len(S_t_w_CP))] = S_t

    t_w_CP = np.arange(0, Tsym + GI, 1 / F_samp)  # new Symbol time includes cyclic prefix

    # plt.figure()
    # plt.plot(t_w_CP, S_t_w_CP)
    # plt.xlabel('Time')
    # plt.ylabel('S(t) with GI')
    # plt.title('OFDM signal with CP in time domain')
    # plt.grid()
    # plt.show()

    # D/A converter:
    # up sample by factor of 5
    # S_t_w_CP_up = signal.resample_poly(S_t_w_CP, 5, 1)
    (S_t_w_CP_up32, t_w_CP_up32) = signal.resample(S_t_w_CP, 32 * len(S_t_w_CP), t_w_CP, domain='time')

    plt.figure()
    plt.plot(t_w_CP_up32, S_t_w_CP_up32)
    plt.xlabel('Time')
    plt.ylabel('S(t) with GI')
    plt.title('upsampled OFDM signal with CP in time domain')
    plt.grid()
    plt.show()

    return S_t_w_CP_up32


def OFDM_FFT_Tx(input_data):
    GI = 0.8 * 1e-6  # 0.8[uS] Long GI
    Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
    Delta_F = 1 / Tsym
    F = 1 / (Tsym + GI)  # analog Frequency
    F_samp = 20 * 1e6  # sample frquency 20MHz

    t_sym = np.arange(0, Tsym, 1 / F_samp)
    F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t_sym))

    S_t_w_CP = np.zeros((len(t_sym) + 16) * Num_Dta_chnk, dtype=np.complex)
    # S_f = np.zeros((Num_Dta_chnk, len(F_axis)), dtype=np.complex)
    for chnk in range(Num_Dta_chnk):
        S_f_chnk = np.zeros(len(F_axis), dtype=np.complex)
        Dta_Tx_chnk = input_data[chnk * 56:(chnk+1) * 56]
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
        # plt.plot(t_sym, S_t_chnk)
        # plt.xlabel('Time')
        # plt.ylabel('S(t)')
        # plt.grid()
        # plt.title('OFDM symbol in time domain')
        # plt.show()

        # Parallel to serial: ?

        # inserting cyclic prefix:
        S_t_chnk_w_CP = np.zeros(int(len(S_t_chnk) + 16), dtype=np.complex)
        CP = S_t_chnk[int(len(S_t_chnk) - 16):len(S_t_chnk)]
        S_t_chnk_w_CP[:16] = CP
        S_t_chnk_w_CP[16:len(S_t_chnk_w_CP)] = S_t_chnk

        t_sym_w_CP = np.arange(0, Tsym + GI, 1 / F_samp)  # new Symbol time includes cyclic prefix

        # plt.figure()
        # plt.plot(t_sym_w_CP, S_t_chnk_w_CP)
        # plt.xlabel('Time')
        # plt.ylabel('S(t) with GI')
        # plt.title('single OFDM symbol with CP in time domain')
        # plt.grid()
        # plt.show()

        S_t_w_CP[chnk*len(S_t_chnk_w_CP):(chnk+1)*len(S_t_chnk_w_CP)] = S_t_chnk_w_CP

    t_w_CP = np.arange(0, (Tsym + GI) * Num_Dta_chnk, 1 / F_samp)
    # plt.figure()
    # plt.plot(t_w_CP, S_t_w_CP)
    # plt.xlabel('Time')
    # plt.ylabel('S(t) with GI')
    # plt.title('OFDM signal with CP in time domain')
    # plt.grid()
    # plt.show()

    # D/A converter:
    # up sample by factor of 32
    # upsample using ZOH interpolation
    up = 32
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

    # to compare symbol energy of up-sampled vs down-sampled signals
    Es = (1/(100*up))*np.sum(np.abs(S_t_w_CP_up[:100*80*up])**2)
    print(Es)

    return S_t_w_CP_up, up



def main():
    # OFDM_Oscilator_Tx()
    OFDM_FFT_Tx(Dta_Tx)


# main()
