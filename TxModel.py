import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal

M = 16  # 16QAM modulator
n = int(1e6)  # data stream length

# Data
rng = np.random.default_rng()

# infase data
m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=n, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))
# quadrature data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=n, dtype=np.int64, endpoint=True) - 1 - int(np.sqrt(M))

data = m_i + 1j * m_q


def TxMod(input_data):
    Tsym = 50 * 1e-9
    m = input_data
    Symst = np.zeros(n, dtype=int)

    # Mapper - converting bit stream to symbols - Gray Coding
    # ind1 = (np.real(m) > 0)
    # ind2 = (np.imag(m) > 0)
    # ind3 = (np.real(m) < 0)
    # ind4 = (np.imag(m) < 0)
    #
    # Symst[np.logical_and(ind1, ind2)] = 0
    # Symst[np.logical_and(ind1, ind4)] = 1
    # Symst[np.logical_and(ind3, ind2)] = 3
    # Symst[np.logical_and(ind3, ind4)] = 2

    # pulse shaping
    # N_samp = 1024
    # alpha = 0.25  # roll off factor
    # Ts = 1
    # Fs = 2  # 2 samples per symbol
    # g_SRRC = np.sqrt(2) * rrcosfilter(N_samp, alpha, Ts, Fs)[1]

    # upsampling
    up = 32
    spaced_m = np.zeros(up * len(m), dtype=np.complex)
    spaced_m[::up] = m
    g_ZOH = np.ones(up, dtype=np.complex)
    m_up = signal.lfilter(g_ZOH, 1, spaced_m)  # add 32 samples NRZ pulse shape
    t_up = np.arange(0, Tsym * n, 1 / (up * Tsym ** -1))
    t_up = t_up[:len(t_up) - 1]

    # plt.plot(t_up[: 16 * up], m_up[: 16 * up])  # print 16 symbols (in time domain)
    # plt.xlabel('Time')
    # plt.ylabel('S(t)')
    # plt.title('transmitted up-sampled ' + str(M) + 'QAM symbol in time domain')
    # plt.grid()
    # plt.show()

    return m_up, up


def main():
    txsig = TxMod(data)[0]


# main()
