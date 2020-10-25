import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

GI = 0.8 * 1e-6  # 0.8[uS] Long GI
Tsym = 3.2 * 1e-6  # 3.2 [uS] symbol time
Delta_F = 1 / Tsym
# F = Delta_F
F = 1 / Tsym  # analog Frequency
# F_samp = 4 * Delta_F  # sample by nyquist
F_samp = 20*1e6  # sample frquency 20MHz
# F_axis = np.arange(-2 * Delta_F, 2 * Delta_F, Delta_F)

t = np.arange(0, Tsym, 1 / F_samp)
F_axis = np.arange(-F_samp / 2, F_samp / 2, F_samp / len(t))
# k = 1
CW_t = {}
CW_f = {}
for k in range(len(F_axis)):
    CW_t[k] = np.exp(1j * 2 * np.pi * (k-len(F_axis)/2) * F * t)
    CW_f[k] = (1 / len(CW_t[k])) * np.fft.fft(CW_t[k])

# plt.plot(t, S_t)
# plt.xlabel('Time')
# plt.ylabel('S(t)')
# plt.plot(F_axis, np.fft.fftshift(S_f11))
plt.figure()
for k in range(len(CW_t)):
    plt.plot(F_axis, np.fft.fftshift(CW_f[k]))
plt.grid()
plt.xlabel('Frequency')
plt.ylabel('S(f)')
plt.show()

print(S_t11)
