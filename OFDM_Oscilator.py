import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

# random 56 symbols of data
rng = np.random.default_rng()

# BPSK
# Dta = 2 * rng.integers(1, high=2, size=56, dtype=np.int64, endpoint=True) - 3

# QPSK
M = 16
# infase data
m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))
# quadrature data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))

Dta = m_i + 1j * m_q

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

# Square Root raised cosine pulse: Not nessesary at this point
N_samp = len(t)
alpha = 0.25  # roll off factor
g_SRRC = rrcosfilter(N_samp, alpha, GI + Tsym, F_samp)[1]
# plt.plot(t, g_SRRC)
# plt.show()
################################################################
# creating the symbols in frequency domain:

skip = 0
for k in range(len(F_axis)):
    if (k - len(F_axis) / 2) < -28 or (k - len(F_axis) / 2) == 0 or (k - len(F_axis) / 2) > 28:
        CW_tk.append(np.zeros(len(t)))
        skip += 1
    else:
        CW_tk.append((1 / np.sqrt(Tsym)) * np.exp(1j * 2 * np.pi * (k - len(F_axis) / 2) * Delta_F * t))
    S_tk.append(Dta[k-skip] * CW_tk[k])


# add summation along the frequency axis:
# this is the signal combined of all the symbols in time:
S_t = np.sum(S_tk, axis=0)

# The signal with all the Symbols in frequency domain:
S_f = (1 / len(S_t)) * np.fft.fft(S_t)

plt.figure(1)
plt.plot(F_axis, np.fft.fftshift(S_f))
plt.xlabel('Frequency')
plt.ylabel('S(f)')
plt.grid()
plt.show()
plt.figure(2)
plt.plot(t, S_t)
plt.xlabel('Time')
plt.ylabel('S(t)')
plt.grid()
plt.show()

print('')
