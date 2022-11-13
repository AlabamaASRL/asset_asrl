import numpy as np
import asset as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

def Ierr(tab,f,xs,ys):
    errs = []
    for x in xs:
        for y in ys:
            errs.append(abs(tab.interp(x,y)-f(x,y)))
            
    return sum(errs)/(len(xs)*len(ys))

vf = ast.VectorFunctions

nx = 11
ny = 11

lim = 1

xse = np.linspace(0,lim,nx)
yse = np.linspace(0,lim,ny)

xsei = np.linspace(0,lim,19*nx)
ysei = np.linspace(0,lim,19*ny)

xsu = [0,.1,.3,.4,.6,.8,.9,1]
xsu = [0,.1,.2,.3,.4,.50,.55,.6,.7,.8,.85,.9,1]
ysu = [0,.1,.2,.3,.4,.6,.7,.8,.9,1]

#ysu =yse

def f(x,y):
    return np.sin(x)*np.cos(y) 
    #return x**3 + y**3 -x*y
def df(x,y):
    return np.array([ 3*x**2 -y ,3*y**2 -x   ])

Xe, Ye = np.meshgrid(xse, yse)
Ze    = f(Xe,Ye)

Xu, Yu = np.meshgrid(xsu, ysu)
Zu    = f(Xu,Yu)

tabe   = ast.VectorFunctions.InterpTable2D(xse,yse,Ze,True)
tabu   = ast.VectorFunctions.InterpTable2D(xsu,ysu,Zu,True)

tabe.vf().rpt([.3,.7],1000000)
tabu.vf().rpt([.3,.7],1000000)

print(tabe.vf().computeall([.3,.7],[1]))

fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"})

surfe = ax1.plot_surface(Xe, Ye, Ze, cmap=cm.viridis,
                       linewidth=0, antialiased=False)


print(tabu.interp(.8,.8))

Xei, Yei = np.meshgrid(xsei, ysei)
Ze    = f(Xei,Yei)

Zei = tabe.interp(Xei,Yei)
Zeu = tabu.interp(Xei,Yei)

fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})

surfei = ax2.plot_surface(Xei, Yei, Zei-Ze, cmap=cm.viridis,
                       linewidth=0, antialiased=False)


print(Ierr(tabe,f,xsei,ysei))
print(Ierr(tabu,f,xsei,ysei))

plt.show()
