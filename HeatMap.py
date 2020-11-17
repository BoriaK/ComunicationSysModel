import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

SymNum = int(10*1e6)

ModNum = 16

Ix = 2 * np.random.random_integers(low=1, high=np.sqrt(ModNum), size=SymNum) - 1 - np.sqrt(ModNum)
Qx = 2 * np.random.random_integers(low=1, high=np.sqrt(ModNum), size=SymNum) - 1 - np.sqrt(ModNum)
X = Ix + 1j * Qx

ni = np.random.normal(loc=0, scale=0.5, size=SymNum)
nq = np.random.normal(loc=0, scale=0.5, size=SymNum)
n = ni + 1j * nq

R = X + n
NOB = 30*ModNum  # Number of Bins in histogram

plt.figure()
plt.hist2d(np.real(R), np.imag(R), bins=NOB, cmap='jet')
plt.show()

# # for heatmap:
#
# H = np.histogram2d(np.real(R), np.imag(R), bins=NOB)
#
# # # flatten the data in H
# h_0 = [item for subh in H[0] for item in subh]
# # # duplicate each value in H[1] = rows of H 10 times
# h_1 = list(np.repeat(H[1][:-1], NOB).round(2))
# # # repeat H[2] = columns of H 10 times
# h_2 = [list(H[2].round(2))[:-1] for j in range(NOB)]
# h_2 = [item for subh2 in h_2 for item in subh2]

# create a dtattable with pandas:
# DF1 = pd.DataFrame({'rows': h_1, 'columns': h_2, 'H': h_0})
# DF1 = DF1.pivot(index='rows', columns='columns', values='H')
# print(DF1)
#
# DF2 = DF1[::-1]
# print(DF2)

# plt.figure()
# sns.heatmap(DF2, cmap='Blues', xticklabels=int(NOB/5), yticklabels=int(NOB/5))
# plt.show()

# for reference:

# plt.scatter(Ix, Qx)
# plt.xlabel('Infase')
# plt.ylabel('Quadrature')
# plt.title('Constellation')
# # plt.legend(['Tx s(t)', 'Rx R(t)'])
# plt.grid()
# plt.colorbar()
# plt.show()
