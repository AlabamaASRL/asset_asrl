import numpy as np
import asset_asrl as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from scipy import interpolate

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments



nx =91
ny =71
lim = np.pi

xs = np.linspace(-lim,lim,nx)
ys = np.linspace(-lim,lim,ny)

xs = list(xs)
ys = list(ys)

xs.pop(6)
ys.pop(9)


xst = np.linspace(-lim,lim,360)
yst = np.linspace(-lim,lim,360)



def f(x,y):return np.cos(x)*np.cos(y)
X, Y = np.meshgrid(xs, ys)
Xt, Yt = np.meshgrid(xst, yst)

Z    = f(X,Y)             

kind = 'cubic' 

Tab2D = vf.InterpTable2D(xs,ys,Z,kind=kind)
Tab2D.ThrowOutOfBounds = True
vf.InterpTable2DSpeedTest(Tab2D.sf(),-lim,lim,-lim,lim,10000000,False)



Ztab = Tab2D.interp(Xt,Yt)
Zex  = f(Xt,Yt)

input("S")


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xt, Yt, Ztab, cmap=cm.viridis,
                       linewidth=1, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
plt.show()

fig1, ax = plt.subplots()
cs = ax.contourf(Xt,Yt,Ztab,levels=400)
cbar = fig1.colorbar(cs)
plt.show()


fig1, ax = plt.subplots()
cs = ax.contourf(Xt,Yt,np.log10(abs(Ztab-Zex) + 1.0e-12),levels=400)
cbar = fig1.colorbar(cs)
plt.show()