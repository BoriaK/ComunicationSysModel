import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

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
CW_t = {}
CW_f = {}
S_t = {}
S_f = {}
X_t = np.sinc(2 * np.pi * F * t)
################################################################
N_samp = len(t)
alpha = 0.25  # roll off factor
# Ts = 1
# Fs = 2  # 2 samples per symbol
g_SRRC = np.sqrt(2) * rrcosfilter(N_samp, alpha, Tsym, F_samp)[1]
# plt.plot(t, g_SRRC)
################################################################
# plt.plot(t, np.sinc(2*np.pi*F*(t-Tsym/2)))
# plt.show()
for k in range(len(F_axis)):
    CW_t[k] = np.exp(1j * 2 * np.pi * (k - len(F_axis) / 2) * Delta_F * t)
    CW_f[k] = (1 / len(CW_t[k])) * np.fft.fft(CW_t[k])
    S_t[k] = X_t * CW_t[k]
    # S_t[k] = g_SRRC * CW_t[k]
    S_f[k] = (1 / len(S_t[k])) * np.fft.fft(S_t[k])

plt.plot(F_axis, np.fft.fftshift(S_f[int(len(F_axis)/2)]))
plt.show()
plt.figure()
# for k in range(len(CW_f)):
#     plt.plot(F_axis, np.fft.fftshift(CW_f[k]))
for k in range(len(S_f)):
    plt.plot(F_axis, np.fft.fftshift(S_f[k]))
plt.grid()
plt.xlabel('Frequency')
plt.ylabel('S(f)')
plt.show()

print('')
