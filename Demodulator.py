import numpy as np


def demapper(R_DataVec, Mod_Num):
    Dta_I = np.zeros(len(R_DataVec), dtype=np.float)
    Dta_Q = np.zeros(len(R_DataVec), dtype=np.float)


    if Mod_Num == 4:
        # the thresholds are {0}

        ind_Re_1 = (np.real(R_DataVec) > 0)
        ind_Re_min1 = (np.real(R_DataVec) < 0)
        ind_Im_1 = (np.imag(R_DataVec) > 0)
        ind_Im_min1 = (np.imag(R_DataVec) < 0)

        Dta_I[ind_Re_1] = 1
        Dta_I[ind_Re_min1] = -1

        Dta_Q[ind_Im_1] = 1
        Dta_Q[ind_Im_min1] = -1

    if Mod_Num == 16:
        # the thresholds are {-2, 0, 2}

        ind_Re_3 = (np.real(R_DataVec) > 2)
        ind_Re_1 = (np.real(R_DataVec) > 0)
        ind_Re_min1 = (np.real(R_DataVec) < 0)
        ind_Re_min3 = (np.real(R_DataVec) < -2)
        ind_Im_3 = (np.imag(R_DataVec) > 2)
        ind_Im_1 = (np.imag(R_DataVec) > 0)
        ind_Im_min1 = (np.imag(R_DataVec) < 0)
        ind_Im_min3 = (np.imag(R_DataVec) < -2)

        Dta_I[ind_Re_3] = 3
        Dta_I[np.logical_and(ind_Re_1, ~ind_Re_3)] = 1
        Dta_I[np.logical_and(~ind_Re_min3, ind_Re_min1)] = -1
        Dta_I[ind_Re_min3] = -3

        Dta_Q[ind_Im_3] = 3
        Dta_Q[np.logical_and(ind_Im_1, ~ind_Im_3)] = 1
        Dta_Q[np.logical_and(~ind_Im_min3, ind_Im_min1)] = -1
        Dta_Q[ind_Im_min3] = -3

    if Mod_Num == 64:
        # the thresholds are {-6, -4, -2, 0, 2, 4, 6}

        ind_Re_7 = (np.real(R_DataVec) > 6)
        ind_Re_5 = (np.real(R_DataVec) > 4)
        ind_Re_3 = (np.real(R_DataVec) > 2)
        ind_Re_1 = (np.real(R_DataVec) > 0)
        ind_Re_min1 = (np.real(R_DataVec) < 0)
        ind_Re_min3 = (np.real(R_DataVec) < -2)
        ind_Re_min5 = (np.real(R_DataVec) < -4)
        ind_Re_min7 = (np.real(R_DataVec) < -6)
        ind_Im_7 = (np.imag(R_DataVec) > 6)
        ind_Im_5 = (np.imag(R_DataVec) > 4)
        ind_Im_3 = (np.imag(R_DataVec) > 2)
        ind_Im_1 = (np.imag(R_DataVec) > 0)
        ind_Im_min1 = (np.imag(R_DataVec) < 0)
        ind_Im_min3 = (np.imag(R_DataVec) < -2)
        ind_Im_min5 = (np.imag(R_DataVec) < -4)
        ind_Im_min7 = (np.imag(R_DataVec) < -6)

        Dta_I[ind_Re_7] = 7
        Dta_I[np.logical_and(ind_Re_5, ~ind_Re_7)] = 5
        Dta_I[np.logical_and(ind_Re_3, ~ind_Re_5)] = 3
        Dta_I[np.logical_and(ind_Re_1, ~ind_Re_3)] = 1
        Dta_I[np.logical_and(~ind_Re_min3, ind_Re_min1)] = -1
        Dta_I[np.logical_and(~ind_Re_min5, ind_Re_min3)] = -3
        Dta_I[np.logical_and(~ind_Re_min7, ind_Re_min5)] = -5
        Dta_I[ind_Re_min7] = -7

        Dta_Q[ind_Im_7] = 7
        Dta_Q[np.logical_and(ind_Im_5, ~ind_Im_7)] = 5
        Dta_Q[np.logical_and(ind_Im_3, ~ind_Im_5)] = 3
        Dta_Q[np.logical_and(ind_Im_1, ~ind_Im_3)] = 1
        Dta_Q[np.logical_and(~ind_Im_min3, ind_Im_min1)] = -1
        Dta_Q[np.logical_and(~ind_Im_min5, ind_Im_min3)] = -3
        Dta_Q[np.logical_and(~ind_Im_min7, ind_Im_min5)] = -5
        Dta_Q[ind_Im_min7] = -7

    Received_Dta = Dta_I + 1j * Dta_Q

    return Received_Dta
