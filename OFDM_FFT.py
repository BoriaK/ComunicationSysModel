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
        S_f[k] = Dta[k - skip]

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
S_t_w_CP = np.zeros(int(len(S_t)+16), dtype=np.complex)
CP = S_t[range(int(len(S_t)-16), len(S_t))]
S_t_w_CP[range(16)] = CP
S_t_w_CP[range(16, len(S_t_w_CP))] = S_t

t_w_CP = np.arange(0, Tsym+GI, 1 / F_samp)  # new Symbol time includes cyclic prefix

# plt.figure()
# plt.plot(t_w_CP, S_t_w_CP)
# plt.xlabel('Time')
# plt.ylabel('S(t) with GI')
# plt.title('OFDM symbol with CP in time domain')
# plt.grid()
# plt.show()

# D/A converter:
# pulse shaping
N_samp = 2
alpha = 0.25  # roll off factor
Ts = Tsym + GI
Fs = F_samp  # 64 samples per symbol
g_SRRC = np.sqrt(2) * rrcosfilter(N_samp, alpha, Ts, Fs)[1]
# up sample by factor of 2
# S_t_w_CP_up = signal.resample_poly(S_t_w_CP, 2, 1)
(S_t_w_CP_up, t_w_CP_up) = signal.resample(S_t_w_CP, 2*len(S_t_w_CP), t_w_CP, domain='time')
S_t_w_CP_SRRC = np.convolve(S_t_w_CP_up, g_SRRC, mode='same')  # add a physical shape to the pulse
(PHY_S_t_w_CP_SRRC, t_w_CP_up_PHY) = signal.resample(S_t_w_CP_SRRC, 16*len(S_t_w_CP_SRRC), t_w_CP_up, domain='time')


plt.figure()
plt.plot(t_w_CP, S_t_w_CP, t_w_CP_up, S_t_w_CP_up, t_w_CP_up_PHY, PHY_S_t_w_CP_SRRC)
plt.xlabel('Time')
plt.ylabel('S(t) with GI')
plt.title('upsampled OFDM symbol with CP and SRRC pulse in time domain')
plt.legend(['S(t)', 'up-sampled S(t)', 'up-sampled S(t) SRRC'])
plt.grid()
plt.show()


