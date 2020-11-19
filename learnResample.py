import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


A = np.array([1, 1, -1, 1, 1, -1, -1, 1, -1, 1])
t = np.arange(start=0, stop=10, step=1)
print(A)

B = np.zeros(2*len(A), dtype=np.float)
B[::2] = A  # spacing A with zeros

h = np.ones(2, dtype=np.float)

A_up_ZOH = signal.lfilter(h, 1, B)
t_up_ZOH = np.arange(start=0, stop=10, step=0.5)

# (A_up, t_up) = signal.resample(A, 4 * len(A), t, domain='time')
# print(A_up)

A_dn = A_up_ZOH[::2]

plt.figure()
# plt.plot(t, A, t_up, A_up, t, A_dn)
plt.plot(t, A, t, A_dn)
# plt.legend(['A', 'A upsampled', 'A downsampled'])
plt.legend(['A', 'A downsampled'])
plt.grid()
plt.show()

