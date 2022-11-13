
import numpy as np
import asset_asrl as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from scipy import interpolate

def F(t):
    return np.array([t**2,t**2.1])


ts = np.linspace(0.0,5.0,120)
ts2 = np.linspace(0,5,1200)
print(F(ts))
#ts = np.array([0, .2,.3, .5,.6,.8,.9,1])

FS = interpolate.interp1d(ts, F(ts),kind='quadratic',axis=1)
FA = ast.VectorFunctions.InterpTable1D(ts,F(ts),axis=1,kind='cubic')

Fe  = F(ts2)

FSi  = FS(ts2)
FAi =FA(ts2)



plt.plot(ts2,abs(Fe[1]-FSi[1]))
plt.plot(ts2,abs(Fe[1]-FAi[1]))

plt.yscale("log")

plt.show()


plt.plot(ts2,abs(Fe[1]))
plt.plot(ts2,abs(FAi[1]))


plt.show()

print(Fe)
print(FSi)
