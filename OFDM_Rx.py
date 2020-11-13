import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

# Tx
# random 56 symbols of data
rng = np.random.default_rng()

# 16QAM
M = 16
# infase data
m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))
# quadrature data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))

Tx_Dta = m_i + 1j * m_q

GI = 0.8 * 1e-6  # 0.8[uS] Long GI
Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
Delta_F = 1 / Tsym
F = 1 / (Tsym + GI)  # analog Frequency
F_samp = 20 * 1e6  # sample frquency 20MHz

t = np.arange(0, Tsym, 1 / F_samp)
F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t))

S_t = np.zeros(len(F_axis), dtype=np.complex)
S_f = np.zeros(len(F_axis), dtype=np.complex)
S_tk = np.zeros(len(F_axis), dtype=np.complex)

# prepare the data in frequency domain:
skip = 0
for k in range(len(F_axis)):
    if (k - len(F_axis) / 2) < -28 or (k - len(F_axis) / 2) == 0 or (k - len(F_axis) / 2) > 28:
        skip += 1
        S_f[k] = 0
    else:
        S_f[k] = Tx_Dta[k - skip]

# prepare time domain signal using IFFT:
S_t = np.fft.ifft(S_f, n=64)

# OFDM symbol in Frequency domain
# plt.figure()
# plt.plot(F_axis, np.abs(S_f))
# plt.xlabel('Frequency')
# plt.ylabel('S(f)')
# plt.grid()
# plt.title('OFDM symbol in frequency domain')
# plt.show()

# OFDM symbol in Time domain
# plt.figure()
# plt.plot(t, S_t)
# plt.xlabel('Time')
# plt.ylabel('S(t)')
# plt.grid()
# plt.title('OFDM symbol in time domain')
# plt.show()

# Parallel to serial: ?

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
# plt.title('OFDM symbol with CP in time domain')
# plt.grid()
# plt.show()

###########################next step#################################
# D/A converter:
# up sample by factor of 5
# S_t_w_CP_up = signal.resample_poly(S_t_w_CP, 5, 1)
# (S_t_w_CP_up32, t_w_CP_up32) = signal.resample(S_t_w_CP, 32 * len(S_t_w_CP), t_w_CP, domain='time')

# plt.figure()
# plt.plot(t_w_CP_up32, S_t_w_CP_up32)
# plt.xlabel('Time')
# plt.ylabel('S(t) with GI')
# plt.title('Transmitted upsampled OFDM symbol with CP in time domain')
# plt.grid()
# plt.show()
##########################################################################################


# Rx

################next step###########################################################
# D/A
# Rx_Sig_w_CP = S_t_w_CP_up32
# Sig_dn_w_CP = signal.decimate(Rx_Sig_w_CP, 32, n=None, ftype='iir', axis=- 1, zero_phase=True)

# plt.figure()
# plt.plot(t_w_CP_up32, Rx_Sig_w_CP)
# plt.xlabel('Time')
# plt.ylabel('S(t) with GI')
# plt.title('Recieved upsampled OFDM symbol with CP in time domain')
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(t_w_CP_up32, S_t_w_CP_up32, t_w_CP_up32, Rx_Sig_w_CP)
# plt.xlabel('Time')
# plt.ylabel('S(t) with GI')
# plt.title('Tx upsampled OFDM symbol Rx upsampled OFDM symbol with CP in time domain')
# plt.legend(['Tx s(t)', 'Rx s(t)'])
# plt.grid()
# plt.show()
###################################################################################

Rx_Sig_w_CP = S_t_w_CP  # Received signal without noise

# plt.figure()
# plt.plot(t_w_CP, S_t_w_CP, t_w_CP, Rx_Sig_w_CP)
# plt.xlabel('Time')
# plt.ylabel('S(t) with GI')
# plt.title('Tx OFDM symbol Rx OFDM symbol with CP in time domain')
# plt.legend(['Tx s(t)', 'Rx s(t)'])
# plt.grid()
# plt.show()

# Add noise Discrete channel
Eb_Discrete = 1
gamma_b_dB_Max = 20
SER_vec = np.zeros(gamma_b_dB_Max + 1, dtype=np.float)
for gamma_b_dB in range(gamma_b_dB_Max + 1):
    gamma_b_L = 10 ** (0.1 * gamma_b_dB)
    N0_Discrete = Eb_Discrete / gamma_b_L
    Ni_Discrete = np.sqrt(N0_Discrete / 2) * np.random.normal(loc=0, scale=1,
                                                              size=len(S_t_w_CP))  # loc = mean, scale = STDV
    Nq_Discrete = np.sqrt(N0_Discrete / 2) * np.random.normal(loc=0, scale=1,
                                                              size=len(S_t_w_CP))  # loc = mean, scale = STDV
    N_Discrete = Ni_Discrete + 1j * Nq_Discrete

    R_t_Disc_w_CP = Rx_Sig_w_CP + N_Discrete

    # plt.figure()
    # plt.plot(t_w_CP, S_t_w_CP, t_w_CP, R_t_Disc_w_CP)
    # plt.xlabel('Time')
    # plt.ylabel('S(t) with GI')
    # plt.title('Tx OFDM symbol Noisy Rx OFDM symbol SNR = ' + str(gamma_b_dB) + ' with CP in time domain')
    # plt.legend(['Tx s(t)', 'Rx R(t)'])
    # plt.grid()
    # plt.show()

    # Remove CP
    Sig_t = R_t_Disc_w_CP[range(16, len(R_t_Disc_w_CP))]

    # plt.figure()
    # plt.plot(t, Sig_t)
    # plt.xlabel('Time')
    # plt.ylabel('S(t)')
    # plt.title('Rx OFDM symbol in time domain')
    # plt.grid()
    # plt.show()

    # S/P Converter

    # FFT Block:
    Sig_f = np.fft.fft(Sig_t, n=64)

    # OFDM symbol in Frequency domain
    # plt.figure()
    # plt.plot(F_axis, np.abs(Sig_f))
    # plt.xlabel('Frequency')
    # plt.ylabel('S(f)')
    # plt.grid()
    # plt.title('Rx OFDM symbol in frequency domain')
    # plt.show()

    # P/S converter
    Dta_vec = np.zeros(56, dtype=np.complex)
    Dta_I = np.zeros(56, dtype=np.float)
    Dta_Q = np.zeros(56, dtype=np.float)

    skip = 0
    for k in range(len(F_axis)):
        if (k - len(F_axis) / 2) < -28 or (k - len(F_axis) / 2) == 0 or (k - len(F_axis) / 2) > 28:
            skip += 1
        else:
            Dta_vec[k - skip] = Sig_f[k]

    # print(Dta_vec)

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
    correct_Symbols = (Tx_Dta == Rx_Dta) * 1
    SER = 1 - np.sum(correct_Symbols) / len(Rx_Dta)

    # print(SER)
    SER_vec[gamma_b_dB] = SER

print(SER_vec)

# plot SER as function of SNR/bit
plt.semilogy(range(gamma_b_dB_Max + 1), SER_vec)
plt.grid()
plt.title('SER as function of SNR/bit')
plt.show()
