import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math

# Tsym = (3.2 + 0.8) * 1e-6  # 3.2 [uS] symbol time + 0.8[uS] Long GI
Tsym = 1
Delta_F = 1 / Tsym
# F = Delta_F
F = 10   # analog Frequency
# F_samp = 4 * Delta_F  # sample by nyquist
F_samp = 20*F  # sample frquency
# F_axis = np.arange(-2 * Delta_F, 2 * Delta_F, Delta_F)

t = np.arange(0, Tsym, 1 / F_samp)
F_axis = np.arange(-F_samp/2, F_samp/2, F_samp/(len(t)-1))
# k = 1
S_t1 = np.exp(-1j * 2 * np.pi * F * t)
S_t2 = np.exp(-1j * 2 * np.pi * (-F) * t)
# S_t = np.cos(2*np.pi*F*t)
S_f1 = (1/len(S_t1))*np.fft.fft(S_t1)
S_f2 = (1/len(S_t2))*np.fft.fft(S_t2)
# plt.plot(t, S_t)
# plt.xlabel('Time')
# plt.ylabel('S(t)')
plt.plot(F_axis, np.fft.fftshift(S_f1), F_axis, np.fft.fftshift(S_f2))
plt.grid()
plt.xlabel('Frequency')
plt.ylabel('S(f)')
plt.show()

print(S_t1)
