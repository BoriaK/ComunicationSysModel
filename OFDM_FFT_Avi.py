import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

# random 56 symbols of data
rng = np.random.default_rng()

# # 16QAM
# M = 16
# # infase data
# m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))
# # quadrature data
# m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))
#
# Dta = m_i + 1j * m_q

# BPSK
Dta = 2 * rng.integers(1, high=2, size=56, dtype=np.int64, endpoint=True) - 3

GI = 0.8 * 1e-6  # 0.8[uS] Long GI
Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
Delta_F = 1 / Tsym
# F = Delta_F
F = 1 / (Tsym + GI)  # analog Frequency
F_samp = 20 * 1e6  # sample frquency 20MHz
# F_axis = np.arange(-2 * Delta_F, 2 * Delta_F, Delta_F)

t = np.arange(0, Tsym, 1 / F_samp)
F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t))
# k = 1
CW_tk = []
CW_fk = []
S_tk = []
S_fk = []

# prepare the data in frequency domain:
skip = 0
for k in range(len(F_axis)):
    if (k - len(F_axis) / 2) < -28 or (k - len(F_axis) / 2) == 0 or (k - len(F_axis) / 2) > 28:
        CW_tk.append(np.zeros(len(t)))
        skip += 1
    else:
        # CW_tk.append((1 / np.sqrt(Tsym)) * np.exp(1j * 2 * np.pi * (k - len(F_axis) / 2) * Delta_F * t))
        CW_tk.append(np.exp(1j * 2 * np.pi * (k - len(F_axis) / 2) * Delta_F * t))
    CW_fk.append((1 / len(F_axis)) * np.fft.fft(CW_tk[k]))
    S_fk.append(np.convolve(Dta[k - skip], CW_fk[k]))

# plt.figure()
# plt.plot(F_axis, np.fft.fftshift(CW_fk))
# plt.xlabel('Frequency')
# plt.ylabel('CW_k(f)')
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(F_axis, np.fft.fftshift(S_fk))
# plt.xlabel('Frequency')
# plt.ylabel('S_k(f)')
# plt.grid()
# plt.show()

# not sure weather it's necessary
S_f = np.sum(S_fk, axis=0)
CW_f = np.sum(CW_fk, axis=0)

plt.figure()
plt.plot(F_axis, np.fft.fftshift(CW_f))
plt.xlabel('Frequency')
plt.ylabel('CW(f)')
plt.grid()
plt.show()

plt.figure()
plt.plot(F_axis, np.abs(np.fft.fftshift(S_f)))
# plt.plot(F_axis, np.abs(S_f))
plt.xlabel('Frequency')
plt.ylabel('S(f)')
plt.grid()
plt.show()

# generating the time OFDM signal:
S_tk = np.fft.ifft(S_fk, axis=-1)
plt.figure()
plt.plot(t, S_tk)
plt.xlabel('Time')
plt.ylabel('S_k(t)')
plt.grid()
plt.show()

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, S_tk[4])
plt.xlabel('Time')
plt.ylabel('S_1(t)')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, S_tk[15])
plt.xlabel('Time')
plt.ylabel('S_5(t)')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, S_tk[25])
plt.xlabel('Time')
plt.ylabel('S_15(t)')
plt.grid()

plt.show()

# combined option:
S_t = np.fft.ifft(S_f)
plt.figure()
plt.plot(t, S_t)
plt.xlabel('Time')
plt.ylabel('S(t)')
plt.grid()
plt.show()
print('')
