import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
from TxModel import TxMod
from scatterPlot import scatter
import math

M = 16  # 16QAM modulator
n = int(1e6)  # data stream length

# Data
rng = np.random.default_rng()

# infase data
m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=n, dtype=np.int64, endpoint=True) - 1 - int(
    np.sqrt(M))
# quadrature data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=n, dtype=np.int64, endpoint=True) - 1 - int(
    np.sqrt(M))

data = m_i + 1j * m_q


def RxMod(TxSignal, up, original_data):
    Tsym = 50 * 1e-9
    F_samp = 20 * 1e6  # sample frquency 20MHz

    t_sym = np.arange(0, Tsym, 1 / F_samp)
    t = np.arange(0, Tsym * n * up, 1 / F_samp)
    t = t[:len(t) - 1]

    Dta_I = np.zeros(n, dtype=np.float)
    Dta_Q = np.zeros(n, dtype=np.float)

    Rx_Sig = TxSignal  # Received signal without noise

    # Add noise continues channel
    Es_Numeric = (1 / (1000 * up)) * np.sum(np.power(np.abs(Rx_Sig[:(1000 * up)]), 2))  # compute average Symbol energy on 1K symbols
    # print(Es_Numeric)
    Eb_Continues = Es_Numeric / np.log2(M)
    gamma_b_dB_Max = 14
    SER_vec = np.zeros(gamma_b_dB_Max + 1, dtype=np.float)
    SER_analitic = np.zeros(gamma_b_dB_Max + 1, dtype=np.float)

    for gamma_b_dB in range(gamma_b_dB_Max + 1):
        gamma_b_L = 10 ** (0.1 * gamma_b_dB)
        N0_Continues = Eb_Continues / gamma_b_L
        Ni_Discrete = np.sqrt(N0_Continues / 2) * np.random.normal(loc=0, scale=1, size=len(Rx_Sig))  # loc = mean,
        # scale = STDV

        Nq_Continues = np.sqrt(N0_Continues / 2) * np.random.normal(loc=0, scale=1, size=len(Rx_Sig))  # loc = mean,
        # scale = STDV

        N_Continues = Ni_Discrete + 1j * Nq_Continues

        R_t = Rx_Sig + N_Continues

        # plt.figure()
        # plt.plot(t[:10 * up], Rx_Sig[:10 * up], t[:10 * up], R_t[:10 * up])
        # plt.xlabel('Time')
        # plt.ylabel('S(t)')
        # plt.title('Clean Rx symbol vs Noisy Rx symbol Eb/N0 = ' + str(gamma_b_dB) + 'dB in time domain')
        # plt.legend(['Tx s(t)', 'Rx R(t)'])
        # plt.grid()
        # plt.show()

        dn = up
        R_t_Disc = R_t[::dn]

        if gamma_b_dB == gamma_b_dB_Max:
            scatter(R_t_Disc, M, gamma_b_dB)

        # De mapper - decision circle

        if M == 4:
            # the thresholds are Real and Im axis
            ind_Re_1 = (np.real(R_t_Disc) > 0)
            ind_Re_min1 = (np.real(R_t_Disc) < 0)
            ind_Im_1 = (np.imag(R_t_Disc) > 0)
            ind_Im_min1 = (np.imag(R_t_Disc) < 0)

            Dta_I[ind_Re_1] = 1
            Dta_I[ind_Re_min1] = -1
            Dta_Q[ind_Im_1] = 1
            Dta_Q[ind_Im_min1] = -1
        elif M == 16:
            # the thresholds are {-2, 0, 2}

            ind_Re_3 = (np.real(R_t_Disc) > 2)
            ind_Re_1 = (np.real(R_t_Disc) > 0)
            ind_Re_min1 = (np.real(R_t_Disc) < 0)
            ind_Re_min3 = (np.real(R_t_Disc) < -2)
            ind_Im_3 = (np.imag(R_t_Disc) > 2)
            ind_Im_1 = (np.imag(R_t_Disc) > 0)
            ind_Im_min1 = (np.imag(R_t_Disc) < 0)
            ind_Im_min3 = (np.imag(R_t_Disc) < -2)

            Dta_I[np.logical_and(ind_Re_1, ind_Re_3)] = 3
            Dta_I[np.logical_and(ind_Re_1, ~ind_Re_3)] = 1
            Dta_I[np.logical_and(~ind_Re_min3, ind_Re_min1)] = -1
            Dta_I[np.logical_and(ind_Re_min3, ind_Re_min1)] = -3

            Dta_Q[np.logical_and(ind_Im_1, ind_Im_3)] = 3
            Dta_Q[np.logical_and(ind_Im_1, ~ind_Im_3)] = 1
            Dta_Q[np.logical_and(~ind_Im_min3, ind_Im_min1)] = -1
            Dta_Q[np.logical_and(ind_Im_min3, ind_Im_min1)] = -3

        Rx_Dta = Dta_I + 1j * Dta_Q

        # performance check:
        Tx_Dta = original_data
        correct_Symbols = (Tx_Dta == Rx_Dta) * 1
        # print(correct_Symbols[:int(len(correct_Symbols) / Num_Dta_chnk)])
        SER = 1 - np.sum(correct_Symbols) / len(Rx_Dta)

        # print(SER)
        SER_vec[gamma_b_dB] = SER
        if M == 4:
            SER_analitic[gamma_b_dB] = math.erfc(np.sqrt(gamma_b_L))
        elif M == 16:
            SER_analitic[gamma_b_dB] = (3 / 2) * math.erfc(np.sqrt(0.4 * gamma_b_L))

    # plot SER as function of SNR/bit
    plt.semilogy(range(gamma_b_dB_Max + 1), SER_vec, range(gamma_b_dB_Max + 1), SER_analitic)
    plt.xlabel('gamma_b[dB]')
    plt.ylabel('SER')
    plt.grid()
    plt.title('SER as function of SNR/bit')
    plt.figlegend(['SER Numeric', 'SER Analytic'])
    plt.show()


def main():
    S_t_up, up = TxMod(data)
    RxMod(S_t_up, up, data)


main()
