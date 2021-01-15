import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math
from OFDM_Tx import OFDM_FFT_Tx
from scatterPlot import scatter
from Auto_Demod import deMapper


# # Tx
# Num_Dta_chnk = int(100 * 1e3)  # number of data chunks
# # random 56 symbols of data per packet
# rng = np.random.default_rng()
#
# # 16QAM
# M = 16
# # infase data
# m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56 * Num_Dta_chnk, dtype=np.int64, endpoint=True) - 1 - int(
#     np.sqrt(M))
# # quadrature data
# m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56 * Num_Dta_chnk, dtype=np.int64, endpoint=True) - 1 - int(
#     np.sqrt(M))
#
# Tx_Dta = m_i + 1j * m_q
#
# # lookup table for Symbol energy discrete model:
# Es_vec = {'2': 1, '4': 2, '16': 10}


# Rx
def OFDM_FFT_Rx(transmitted_signal, up, original_data, Mod_Num):
    Num_Dta_chnk = int((len(transmitted_signal) / up)*0.8 / 64)
    GI = 0.8 * 1e-6  # 0.8[uS] Long GI
    Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
    Delta_F = 1 / Tsym
    F = 1 / (Tsym + GI)  # analog Frequency
    F_samp = 20 * 1e6  # sample frquency 20MHz

    t_sym = np.arange(0, Tsym, 1 / F_samp)
    t_w_CP = np.arange(0, (Tsym + GI) * Num_Dta_chnk, 1 / F_samp)
    F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t_sym))

    Dta_vec = np.zeros(56 * Num_Dta_chnk, dtype=np.complex)

    Rx_Sig_w_CP = transmitted_signal  # Received signal without noise

    # Add noise Discrete channel
    # Es_Numeric = (1 / (100*up)) * np.sum(np.abs(Rx_Sig_w_CP[:100*up*80]) ** 2)  # compute average Symbol energy on 100
    # symbols, 80 samples per symbol upsampled by factor up
    # print(Es_Numeric)
    # lookup table for Symbol energy discrete model:
    Es_vec = {'2': 1, '4': 2, '16': 10, '64': 42}
    Es_Theoretical = Es_vec[str(Mod_Num)]
    Eb_Discrete = Es_Theoretical / np.log2(Mod_Num)
    gamma_b_dB_Max = 15
    SER_vec = np.zeros(gamma_b_dB_Max + 1, dtype=np.float)
    SER_analitic = np.zeros(gamma_b_dB_Max + 1, dtype=np.float)
    for gamma_b_dB in range(gamma_b_dB_Max + 1):
        # for gamma_b_dB in range(gamma_b_dB_Max, gamma_b_dB_Max + 1):    # for debug for single SNR/bit value
        gamma_b_L = 10 ** (0.1 * gamma_b_dB)
        N0_Discrete = Eb_Discrete / gamma_b_L
        Pn = (N0_Discrete / 2) * (1 / 64)  # the poise power for 1 symbol devided by the number of samples
        Ni_Discrete = np.sqrt(Pn) * np.random.normal(loc=0, scale=1,
                                                     size=len(Rx_Sig_w_CP))  # loc = mean, scale = STDV
        Nq_Discrete = np.sqrt(Pn) * np.random.normal(loc=0, scale=1,
                                                     size=len(Rx_Sig_w_CP))  # loc = mean, scale = STDV
        N_Discrete = Ni_Discrete + 1j * Nq_Discrete

        R_t_w_CP = Rx_Sig_w_CP + N_Discrete

        # plt.figure()
        # plt.plot(t_w_CP_up[:80*up], Rx_Sig_w_CP[:80*up], t_w_CP_up[:80*up], R_t_w_CP[:80*up])
        # plt.xlabel('Time')
        # plt.ylabel('S(t) with GI')
        # plt.title('Clean Rx OFDM symbol vs Noisy Rx OFDM symbol Eb/N0 = ' + str(gamma_b_dB) + 'dB with CP in time domain')
        # plt.legend(['Tx s(t)', 'Rx R(t)'])
        # plt.grid()
        # plt.show()

        # A/D
        dn = up
        R_t_Disc_w_CP = R_t_w_CP[::dn]

        # S/P Converter
        for chnk in range(Num_Dta_chnk):
            R_t_Disc_chnk_w_CP = R_t_Disc_w_CP[int(chnk * (len(R_t_Disc_w_CP) / Num_Dta_chnk)):
                                               int((chnk + 1) * (len(R_t_Disc_w_CP) / Num_Dta_chnk))]

            # Remove CP
            Sig_t_chnk = R_t_Disc_chnk_w_CP[16:len(R_t_Disc_chnk_w_CP)]
            # plt.figure()
            # plt.plot(t_sym, Sig_t_chnk)
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

            Dta_vec[chnk * len(Dta_vec_chnk):(chnk + 1) * len(Dta_vec_chnk)] = Dta_vec_chnk

        # constalation:
        if gamma_b_dB == gamma_b_dB_Max:
            scatter(Dta_vec, Mod_Num, gamma_b_dB)

        # Demapper - descision circle

        Rx_Dta = deMapper(Dta_vec, Mod_Num)

        # print(Dta_vec)

        # performance check:
        Tx_Dta = original_data
        correct_Symbols = (Tx_Dta == Rx_Dta) * 1
        SER = 1 - np.sum(correct_Symbols) / len(Rx_Dta)

        # print(SER)
        SER_vec[gamma_b_dB] = SER
        SER_analitic[gamma_b_dB] = 1 - (1 - ((np.sqrt(Mod_Num) - 1) / np.sqrt(Mod_Num)) *
                                        math.erfc(
                                            np.sqrt((3 / (Mod_Num - 1)) * 0.5 * np.log2(Mod_Num) * gamma_b_L))) ** 2

    # print(SER_vec)
    # print(SER_analitic)

    # plot SER as function of SNR/bit
    plt.semilogy(range(gamma_b_dB_Max + 1), SER_vec, range(gamma_b_dB_Max + 1), SER_analitic)
    plt.xlabel('gamma_b[dB]')
    plt.ylabel('SER')
    plt.grid()
    plt.title('SER as function of SNR/bit')
    plt.figlegend(['SER Numeric', 'SER Analytic'])
    plt.show()

# def main():
#     S_t_w_CP_up, up = OFDM_FFT_Tx(Tx_Dta)
#     OFDM_FFT_Rx(S_t_w_CP_up, up, Tx_Dta)


# main()
