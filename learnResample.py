import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


A = [1, 1, -1, 1, 1, -1, -1, 1, -1, 1]
t = np.arange(start=0, stop=10, step=1)
print(A)

(A_up, t_up) = signal.resample(A, 4 * len(A), t, domain='time')
print(A_up)

A_dn = signal.decimate(A_up, 4, n=None, ftype='iir', axis=- 1, zero_phase=True)

plt.figure()
# plt.plot(t, A, t_up, A_up, t, A_dn)
plt.plot(t, A, t, A_dn)
# plt.legend(['A', 'A upsampled', 'A downsampled'])
plt.legend(['A', 'A downsampled'])
plt.grid()
plt.show()

