
import numpy as np
import asset as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from scipy import interpolate

def F(t):
    return np.array([t**2,np.cos(t)])


ts = np.linspace(0.0,5.0,80)
ts2 = np.linspace(0,5,520)

#ts = np.array([0, .2,.3, .5,.6,.8,.9,1])

FS = interpolate.interp1d(ts, F(ts),kind='quadratic',axis=1)
FA = ast.VectorFunctions.InterpTable1D(ts,F(ts),True)

Fe  = F(ts2)
FSi  = FS(ts2)
FAi = np.array([FA.interp(t) for t in ts2]).T



plt.plot(ts2,abs(Fe[1]-FSi[1]))
plt.plot(ts2,abs(Fe[1]-FAi[1]))

plt.yscale("log")

plt.show()


plt.plot(ts2,abs(Fe[1]))
plt.plot(ts2,abs(FAi[1]))


plt.show()

print(Fe)
print(FSi)
