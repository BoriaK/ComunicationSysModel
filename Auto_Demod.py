import numpy as np


def deMapper(R_DataVec, Mod_Num):
    Dta_I = np.zeros(len(R_DataVec), dtype=np.float)
    Dta_Q = np.zeros(len(R_DataVec), dtype=np.float)

    N = np.sqrt(Mod_Num) - 1  # number of total thresholds
    ThrVec = np.arange(-(N - 1), N, 2)  # threshold vec for I and Q

    # creates an empty table  of indices for values of received I and Q in regards to thresholds
    IndsRe = np.zeros((int(np.sqrt(Mod_Num)), len(R_DataVec)), dtype=bool)
    IndsIm = np.zeros((int(np.sqrt(Mod_Num)), len(R_DataVec)), dtype=bool)

    # for all the indexes in the middle range:
    # fill in the negative values first:
    for i in range(int(np.sqrt(Mod_Num) / 2)):
        IndsRe[i] = (np.real(R_DataVec) < ThrVec[i])
        IndsIm[i] = (np.imag(R_DataVec) < ThrVec[i])
    # fill in the positive values after the negative:
    for i in range(int(np.sqrt(Mod_Num) / 2), int(np.sqrt(Mod_Num))):
        IndsRe[i] = (np.real(R_DataVec) > ThrVec[i - 1])
        IndsIm[i] = (np.imag(R_DataVec) > ThrVec[i - 1])

    # make the decision based on ML
    # the edge cases are straight forward hard decision:
    Dta_I[IndsRe[0]] = ThrVec[0] - 1
    Dta_Q[IndsIm[0]] = ThrVec[0] - 1
    Dta_I[IndsRe[-1]] = ThrVec[-1] + 1
    Dta_Q[IndsIm[-1]] = ThrVec[-1] + 1
    for i in range(1, int(np.sqrt(Mod_Num) / 2)):
        Dta_I[np.logical_and(~IndsRe[i - 1], IndsRe[i])] = ThrVec[i] - 1
        Dta_Q[np.logical_and(~IndsIm[i - 1], IndsIm[i])] = ThrVec[i] - 1

    for i in range(int(np.sqrt(Mod_Num) / 2), int(np.sqrt(Mod_Num) - 1)):
        Dta_I[np.logical_and(IndsRe[i], ~IndsRe[i + 1])] = ThrVec[i - 1] + 1
        Dta_Q[np.logical_and(IndsIm[i], ~IndsIm[i + 1])] = ThrVec[i - 1] + 1

    Received_Dta = Dta_I + 1j * Dta_Q

    return Received_Dta
