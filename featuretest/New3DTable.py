import numpy as np
import asset_asrl as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator


np.set_printoptions(linewidth=120)
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments



nx =95
ny =96
nz =97

lim = np.pi

xs = np.linspace(-lim,lim,nx)
ys = np.linspace(-lim,lim,ny)
zs = np.linspace(-lim,lim,ny)

def f(x,y,z):return np.cos(x)*np.cos(y)*np.cos(z)


X, Y, Z = np.meshgrid(xs, ys, zs,indexing = 'ij')

Fs = f(X,Y,Z)

Tab = vf.InterpTable3D(xs,ys,zs,Fs,kind='linear')

STab = RegularGridInterpolator((xs,ys,zs),Fs,method="linear")


px =1.9
py =.4
pz = .222

print(Tab(px,py,pz) - STab([px,py,pz]))

