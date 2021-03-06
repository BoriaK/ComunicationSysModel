import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

# random 56 symbols of data
rng = np.random.default_rng()

# 16QAM
M = 16
# infase data
m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))
# quadrature data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))

Dta = m_i + 1j * m_q

GI = 0.8 * 1e-6  # 0.8[uS] Long GI
Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
Delta_F = 1 / Tsym
F = 1 / (Tsym + GI)  # analog Frequency
F_samp = 20 * 1e6  # sample frquency 20MHz

t = np.arange(0, Tsym, 1 / F_samp)
F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t))

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
    S_tk[k][:] = Dta[k - skip]*CW_tk[k]

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
S_t_w_CP = np.zeros(int(len(S_t)+16), dtype=np.complex)
CP = S_t[range(int(len(S_t)-16), len(S_t))]
S_t_w_CP[range(16)] = CP
S_t_w_CP[range(16, len(S_t_w_CP))] = S_t

t_w_CP = np.arange(0, Tsym+GI, 1 / F_samp)  # new Symbol time includes cyclic prefix


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
(S_t_w_CP_up, S_t_w_CP) = signal.resample(S_t_w_CP, 5*len(S_t_w_CP), t_w_CP, domain='time')

# plt.figure()
# plt.plot(S_t_w_CP, S_t_w_CP_up)
# plt.xlabel('Time')
# plt.ylabel('S(t) with GI')
# plt.title('upsampled OFDM signal with CP in time domain')
# plt.grid()
# plt.show()

############################### for debug:#################################
# inserting cyclic prefix in all the carriers sepperatly:
# S_tk_w_CP = np.zeros((len(F_axis), int(len(S_t)+16)), dtype=np.complex)
# CPk = S_tk[:, range(int(len(S_t)-16), len(S_t))]
# S_tk_w_CP[:, range(16)] = CPk
# S_tk_w_CP[:, range(16, int(len(S_t)+16))] = S_tk
#
# # Plot some Carriers of the OFDM symbol with CP in time
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(t_w_CP, S_tk_w_CP[4])
# plt.xlabel('Time')
# plt.ylabel('S_4(t) + CP')
# plt.grid()
#
# plt.subplot(3, 1, 2)
# plt.plot(t_w_CP, S_tk_w_CP[15])
# plt.xlabel('Time')
# plt.ylabel('S_15(t) + CP')
# plt.grid()
#
# plt.subplot(3, 1, 3)
# plt.plot(t_w_CP, S_tk_w_CP[25])
# plt.xlabel('Time')
# plt.ylabel('S_25(t) + CP')
# plt.grid()
# plt.suptitle('OFDM symbol with CP in time domain')
# plt.show()

############################################################################
