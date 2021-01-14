import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math
from scatterPlot import scatter
from Demodulator import demapper
from OFDM_Tx import OFDM_FFT_Tx
from OFDM_Rx import OFDM_FFT_Rx

Num_Dta_chnk = int(100 * 1e3)  # number of data chunks
# random 56 symbols of data per packet
rng = np.random.default_rng()
# M-QAM
# M = {4, 16, 64}
M = 64
# infase data
m_i = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56 * Num_Dta_chnk, dtype=np.int64, endpoint=True) - 1 - int(
    np.sqrt(M))
# quadrature data
m_q = 2 * rng.integers(1, high=int(np.sqrt(M)), size=56 * Num_Dta_chnk, dtype=np.int64, endpoint=True) - 1 - int(
    np.sqrt(M))

Dta_Tx = m_i + 1j * m_q


def main():
    S_t_w_CP_up, up = OFDM_FFT_Tx(Dta_Tx)
    OFDM_FFT_Rx(S_t_w_CP_up, up, Dta_Tx, M)


main()
