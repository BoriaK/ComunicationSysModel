import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math
from OFDM_Tx import OFDM_FFT_Tx

# Tx
Num_Dta_chnk = int(1e3)  # number of data chunks
# random 56 symbols of data per packet
rng = np.random.default_rng()

# 16QAM
M = 16
# infase data
m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56 * Num_Dta_chnk, dtype=np.int64, endpoint=True) - 1 - int(
    np.sqrt(M))
# quadrature data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56 * Num_Dta_chnk, dtype=np.int64, endpoint=True) - 1 - int(
    np.sqrt(M))

Tx_Dta = m_i + 1j * m_q


# Rx
def OFDM_FFT_Rx(received_signal, original_data, transmitted_signal):
    GI = 0.8 * 1e-6  # 0.8[uS] Long GI
    Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
    Delta_F = 1 / Tsym
    F = 1 / (Tsym + GI)  # analog Frequency
    F_samp = 20 * 1e6  # sample frquency 20MHz

    t_sym = np.arange(0, Tsym, 1 / F_samp)
    t_w_CP = np.arange(0, (Tsym + GI) * Num_Dta_chnk, 1 / F_samp)
    F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t_sym))

    Dta_vec = np.zeros(56 * Num_Dta_chnk, dtype=np.complex)
    Dta_I = np.zeros(56 * Num_Dta_chnk, dtype=np.float)
    Dta_Q = np.zeros(56 * Num_Dta_chnk, dtype=np.float)

    Rx_Sig_w_CP = received_signal  # Received signal without noise
    if len(Rx_Sig_w_CP) > len(t_w_CP):
        ################next step###########################################################
        # A/D
        # plt.figure()
        # plt.plot(Rx_Sig_w_CP[range(int(len(Rx_Sig_w_CP) / Num_Dta_chnk))])
        # plt.xlabel('Time')
        # plt.ylabel('S(t) with GI')
        # plt.title('Recieved upsampled OFDM symbol with CP in time domain')
        # plt.grid()
        # plt.show()

        Sig_dn_w_CP = signal.decimate(Rx_Sig_w_CP, 32, n=None, ftype='iir', axis=- 1, zero_phase=True)
        ###################################################################################

    else:
        Sig_dn_w_CP = Rx_Sig_w_CP

    plt.figure()
    plt.plot(t_w_CP[range(int(len(t_w_CP) / Num_Dta_chnk))],
             transmitted_signal[range(int(len(transmitted_signal) / Num_Dta_chnk))],
             t_w_CP[range(int(len(t_w_CP) / Num_Dta_chnk))], Sig_dn_w_CP[range(int(len(Sig_dn_w_CP) / Num_Dta_chnk))])
    plt.xlabel('Time')
    plt.ylabel('S(t) with GI')
    plt.title('perliminary Tx OFDM symbol vs Down sampled Rx OFDM symbol with CP in time domain')
    plt.legend(['Tx s(t)', 'Rx s(t)'])
    plt.grid()
    plt.show()

    # Add noise Discrete channel
    Eb_Discrete = 1
    gamma_b_dB_Max = 40
    SER_vec = np.zeros(gamma_b_dB_Max + 1, dtype=np.float)
    # for gamma_b_dB in range(gamma_b_dB_Max + 1):
    for gamma_b_dB in range(gamma_b_dB_Max, gamma_b_dB_Max + 1):    # for debug for single SNR/bit value
        gamma_b_L = 10 ** (0.1 * gamma_b_dB)
        N0_Discrete = Eb_Discrete / gamma_b_L
        Ni_Discrete = np.sqrt(N0_Discrete / 2) * np.random.normal(loc=0, scale=1,
                                                                  size=len(Sig_dn_w_CP))  # loc = mean, scale = STDV
        Nq_Discrete = np.sqrt(N0_Discrete / 2) * np.random.normal(loc=0, scale=1,
                                                                  size=len(Sig_dn_w_CP))  # loc = mean, scale = STDV
        N_Discrete = Ni_Discrete + 1j * Nq_Discrete

        R_t_Disc_w_CP = Sig_dn_w_CP + N_Discrete

        # plt.figure()
        # plt.plot(t_w_CP, S_t_w_CP, t_w_CP, R_t_Disc_w_CP)
        # plt.xlabel('Time')
        # plt.ylabel('S(t) with GI')
        # plt.title('Tx OFDM symbol Noisy Rx OFDM symbol SNR = ' + str(gamma_b_dB) + ' with CP in time domain')
        # plt.legend(['Tx s(t)', 'Rx R(t)'])
        # plt.grid()
        # plt.show()

        # S/P Converter
        for chnk in range(Num_Dta_chnk):
            R_t_Disc_chnk_w_CP = R_t_Disc_w_CP[range(int(chnk * (len(R_t_Disc_w_CP) / Num_Dta_chnk)),
                                                     int((chnk + 1) * (len(R_t_Disc_w_CP) / Num_Dta_chnk)))]

            # Remove CP
            Sig_t_chnk = R_t_Disc_chnk_w_CP[range(16, len(R_t_Disc_chnk_w_CP))]
            # plt.figure()
            # plt.plot(t, Sig_t)
            # plt.xlabel('Time')
            # plt.ylabel('S(t)')
            # plt.title('single Rx OFDM symbol in time domain')
            # plt.grid()
            # plt.show()

            # FFT Block:
            Sig_f_chnk = np.fft.fftshift(np.fft.fft(Sig_t_chnk, n=64))

            # OFDM symbol in Frequency domain
            # plt.figure()
            # plt.plot(F_axis, np.abs(Sig_f_chnk))
            # plt.xlabel('Frequency')
            # plt.ylabel('S(f)')
            # plt.grid()
            # plt.title('Rx OFDM symbol in frequency domain')
            # plt.show()

            # P/S converter
            Dta_vec_chnk = np.zeros(56, dtype=np.complex)
            skip = 0
            for k in range(len(F_axis)):
                if (k - len(F_axis) / 2) < -28 or (k - len(F_axis) / 2) == 0 or (k - len(F_axis) / 2) > 28:
                    skip += 1
                else:
                    Dta_vec_chnk[k - skip] = Sig_f_chnk[k]

                # print(Dta_vec_chnk)

            Dta_vec[range(chnk * len(Dta_vec_chnk), (chnk + 1) * len(Dta_vec_chnk))] = Dta_vec_chnk

        # Demapper - descision circle

        # the thresholds are {-2, 0, 2}

        ind_Re_3 = (np.real(Dta_vec) > 2)
        ind_Re_1 = (np.real(Dta_vec) > 0)
        ind_Re_min1 = (np.real(Dta_vec) < 0)
        ind_Re_min3 = (np.real(Dta_vec) < -2)
        ind_Im_3 = (np.imag(Dta_vec) > 2)
        ind_Im_1 = (np.imag(Dta_vec) > 0)
        ind_Im_min1 = (np.imag(Dta_vec) < 0)
        ind_Im_min3 = (np.imag(Dta_vec) < -2)

        Dta_I[np.logical_and(ind_Re_1, ind_Re_3)] = 3
        Dta_I[np.logical_and(ind_Re_1, ~ind_Re_3)] = 1
        Dta_I[np.logical_and(~ind_Re_min3, ind_Re_min1)] = -1
        Dta_I[np.logical_and(ind_Re_min3, ind_Re_min1)] = -3

        Dta_Q[np.logical_and(ind_Im_1, ind_Im_3)] = 3
        Dta_Q[np.logical_and(ind_Im_1, ~ind_Im_3)] = 1
        Dta_Q[np.logical_and(~ind_Im_min3, ind_Im_min1)] = -1
        Dta_Q[np.logical_and(ind_Im_min3, ind_Im_min1)] = -3

        Rx_Dta = Dta_I + 1j * Dta_Q

        # print(Dta_vec)

        # performance check:
        Tx_Dta = original_data
        correct_Symbols = (Tx_Dta == Rx_Dta) * 1
        print(correct_Symbols[range(int(len(correct_Symbols) / Num_Dta_chnk))])
        SER = 1 - np.sum(correct_Symbols) / len(Rx_Dta)

        # print(SER)
        SER_vec[gamma_b_dB] = SER

    # print(SER_vec)

    # plot SER as function of SNR/bit
    plt.semilogy(range(gamma_b_dB_Max + 1), SER_vec)
    plt.xlabel('gamma_b[dB]')
    plt.ylabel('SER')
    plt.grid()
    plt.title('SER as function of SNR/bit')
    plt.show()


def main():
    S_t_w_CP_up32, S_t_w_CP = OFDM_FFT_Tx(Tx_Dta)
    OFDM_FFT_Rx(S_t_w_CP_up32, Tx_Dta, S_t_w_CP)


main()
