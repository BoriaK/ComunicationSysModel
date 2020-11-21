import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


A = np.array([1, 1, 1, 1, 1])
t = np.arange(start=0, stop=5, step=1)
print(A)

up = 5

B = np.zeros(up*len(A), dtype=np.float)
B[2:len(B):up] = A  # spacing A with zeros, with a 2 sample delay

h = np.ones(up, dtype=np.float)

A_up_ZOH = signal.lfilter(h, 1, B)
t_up_ZOH = np.arange(start=0, stop=len(A), step=1/up)

# (A_up, t_up) = signal.resample(A, 4 * len(A), t, domain='time')
# print(A_up)

A_dn = A_up_ZOH[::up]

plt.figure()
# plt.plot(t, A, t_up_ZOH, A_up_ZOH, t, A_dn)
plt.plot(t_up_ZOH, B, t_up_ZOH, A_up_ZOH)
# plt.plot(t, A, t, A_dn)
# plt.legend(['A', 'A upsampled', 'A downsampled'])
plt.legend(['A_delta', 'A upsampled'])
# plt.legend(['A', 'A downsampled'])
plt.grid()
plt.show()

