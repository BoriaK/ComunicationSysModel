import numpy as np
from matplotlib import pyplot as plt

# SymNum = int(10 * 1e6)
#
# ModNum = 16
#
# Ix = 2 * np.random.random_integers(low=1, high=np.sqrt(ModNum), size=SymNum) - 1 - np.sqrt(ModNum)
# Qx = 2 * np.random.random_integers(low=1, high=np.sqrt(ModNum), size=SymNum) - 1 - np.sqrt(ModNum)
# X = Ix + 1j * Qx
#
# ni = np.random.normal(loc=0, scale=0.5, size=SymNum)
# nq = np.random.normal(loc=0, scale=0.5, size=SymNum)
# n = ni + 1j * nq
#
# R = X + n
# NOB = 30 * ModNum  # Number of Bins in histogram


def scatter(InputSig, ModNum, SNR_bit):
    NOB = 30 * ModNum  # Number of Bins in histogram
    plt.figure()
    plt.hist2d(np.real(InputSig), np.imag(InputSig), bins=NOB, cmap='jet')
    plt.xlabel('Infase')
    plt.ylabel('Quadrature')
    plt.title('Constellation of Rx OFDM signal ' + str(ModNum) + '-QAM ' + 'with SNR = ' + str(SNR_bit) + 'dB')
    plt.grid()
    plt.show()
