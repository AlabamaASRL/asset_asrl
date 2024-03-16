import numpy as np
import asset_asrl as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

import scipy 
import time





np.set_printoptions(linewidth=120)
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments



nx =100
ny =100
nz =100

lim = np.pi

xs = np.linspace(-lim,lim,nx)
ys = np.linspace(-lim,lim,ny)
zs = np.linspace(-lim,lim,nz)



def f(x,y,z):
    return np.cos(x)*np.cos(y)*np.cos(z) + z**3
def df(x,y,z):
    return - np.array([np.sin(x)*np.cos(y)*np.cos(z),
                       np.cos(x)*np.sin(y)*np.cos(z),
                       np.cos(x)*np.cos(y)*np.sin(z) - 3*(z**2)])
def d2f(x,y,z):
    row1 = - np.array([ np.cos(x)*np.cos(y)*np.cos(z),-np.sin(x)*np.sin(y)*np.cos(z),-np.sin(x)*np.cos(y)*np.sin(z)])
    row2 = - np.array([-np.sin(x)*np.sin(y)*np.cos(z), np.cos(x)*np.cos(y)*np.cos(z),-np.cos(x)*np.sin(y)*np.sin(z)])
    row3 = - np.array([-np.sin(x)*np.cos(y)*np.sin(z),-np.cos(x)*np.sin(y)*np.sin(z), np.cos(x)*np.cos(y)*np.cos(z) - 6*z])
    return  np.array([row1,row2,row3])


X, Y, Z = np.meshgrid(xs, ys, zs,indexing = 'ij')
Fs = f(X,Y,Z)


Tab1 = vf.InterpTable3D(xs,ys,zs,Fs,kind='cubic')


Tab2 = vf.InterpTable3D(xs,ys,zs,Fs,kind='cubic',cache=True)


Tab3 = vf.InterpTable3D(xs,ys,zs,Fs,kind='cubic')
Tab3.FastProduct = True


vf.InterpTable3DSpeedTest(Tab1.sf(),-lim,lim,-lim,lim,-lim,lim,1000000,False)
vf.InterpTable3DSpeedTest(Tab2.sf(),-lim,lim,-lim,lim,-lim,lim,1000000,False)
vf.InterpTable3DSpeedTest(Tab3.sf(),-lim,lim,-lim,lim,-lim,lim,1000000,False)

