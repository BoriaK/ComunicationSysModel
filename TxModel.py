import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal

M = 4  # Q-PSK modulator
n = int(1e6)  # data stream length
Symst = np.zeros(n, dtype=int)
# Data

rng = np.random.default_rng()

m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=int(n), dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))  #
# infase data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=int(n), dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(
    M))  # quadrature data
m = m_i + 1j * m_q

# converting bit stream to symbols - Gray Coding
ind1 = (np.real(m) > 0)
ind2 = (np.imag(m) > 0)
ind3 = (np.real(m) < 0)
ind4 = (np.imag(m) < 0)

Symst[np.logical_and(ind1, ind2)] = 0
Symst[np.logical_and(ind1, ind4)] = 1
Symst[np.logical_and(ind3, ind2)] = 3
Symst[np.logical_and(ind3, ind4)] = 2

# pulse shaping
N_samp = 1024
alpha = 0.25  # roll off factor
Ts = 1
Fs = 2  # 2 samples per symbol
g_SRRC = np.sqrt(2)*rrcosfilter(N_samp, alpha, Ts, Fs)[1]

# upsampling
m_upS = signal.resample(m, 2*n)   # upsampled signal by factor of 2 (1 samples per symbol)
m_SRRC = np.convolve(m_upS, g_SRRC, mode='same')  # add a physical shape to the pulse
m_SRRC = signal.resample(m_SRRC, 16*n)  # upsample the waveform to 16 samples per symbol

# plt.plot(m_SRRC[range(0, 8*16)])   # print 8 symbols (in time domain)

Es = (1/(100*16))*np.sum(np.power(np.abs(m_SRRC[range(0, 100*16)]), 2))
Eb = (1/np.log2(M))*Es







