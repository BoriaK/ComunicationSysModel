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

# D/A converter:
# up sample by factor of 5
# S_t_w_CP_up = signal.resample_poly(S_t_w_CP, 5, 1)
(S_t_w_CP_up, t_w_CP_up) = signal.resample(S_t_w_CP, 5 * len(S_t_w_CP), t_w_CP, domain='time')

# plt.figure()
# plt.plot(t_w_CP_up, S_t_w_CP_up)
# plt.xlabel('Time')
# plt.ylabel('S(t) with GI')
# plt.title('upsampled OFDM symbol with CP in time domain')
# plt.grid()
# plt.show()

# Rx
Rx_Sig_w_CP = S_t_w_CP_up
Sig_dn_w_CP = signal.decimate(Rx_Sig_w_CP, 5, n=None, ftype='iir', axis=- 1, zero_phase=True)

# plt.figure()
# plt.plot(t_w_CP, S_t_w_CP, t_w_CP, Sig_dn_w_CP)
# plt.xlabel('Time')
# plt.ylabel('S(t) with GI')
# plt.title('Tx OFDM symbol Rx OFDM symbol with CP in time domain')
# plt.grid()
# plt.show()

# Remove CP
Sig_t = Sig_dn_w_CP[range(16, len(Sig_dn_w_CP))]

# plt.figure()
# plt.plot(t, Sig_t)
# plt.xlabel('Time')
# plt.ylabel('S(t)')
# plt.title('Rx OFDM symbol in time domain')
# plt.grid()
# plt.show()

# FFT Block:
Sig_f = np.fft.fft(Sig_t, n=64)

# OFDM symbol in Frequency domain
plt.figure()
plt.plot(F_axis, np.abs(Sig_f))
plt.xlabel('Frequency')
plt.ylabel('S(f)')
plt.grid()
plt.title('Rx OFDM symbol in frequency domain')
plt.show()


