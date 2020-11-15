import numpy as np
from matplotlib import pyplot as plt
from commpy.filters import rrcosfilter
from scipy import signal
import math
import seaborn as sns


Ix = 2 * np.random.random_integers(low=1, high=2, size=1000) - 1 - 2
Qx = 2 * np.random.random_integers(low=1, high=2, size=1000) - 1 - 2
X = Ix + 1j * Qx

ni = np.random.normal(loc=0, scale=0.5, size=1000)
nq = np.random.normal(loc=0, scale=0.5, size=1000)
n = ni + 1j * nq

R = X + n
# X, Y = np.meshgrid(Ix, Qx)
# print(Ix)
# print(Qx)
h = np.histogram2d(np.real(R), np.imag(R), bins=121)
print(h)
# print(np.sum(h, axis=0))
plt.figure()
# sns.heatmap(h, xticklabels=np.arange(start=-1.5, stop=1.5+0.025, step=0.025), yticklabels=np.arange(start=1.5, stop=(-1.5-0.025), step=-0.025))
sns.heatmap(h[0], xticklabels=h[1], yticklabels=h[2])
# sns.heatmap(h)
plt.show()



# plt.scatter(np.real(X), np.imag(X), c=h)
plt.scatter(Ix, Qx)
plt.xlabel('Infase')
plt.ylabel('Quadrature')
plt.title('Constellation')
# plt.legend(['Tx s(t)', 'Rx R(t)'])
plt.grid()
plt.colorbar()
plt.show()
