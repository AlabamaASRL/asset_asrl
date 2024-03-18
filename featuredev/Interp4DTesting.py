import numpy as np
import asset_asrl as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

import scipy 
import time

from asset_asrl.VectorFunctions.Extensions.DerivChecker import FDDerivChecker



def f(x,y,z,w):
    return np.cos(x)*np.cos(y)*np.cos(z)*np.cos(w) + w**4 + 1/(1+ x**2 + y**2 + z**2 + w**2)



np.set_printoptions(linewidth=120)
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments



if __name__ == "__main__":

    nx =30
    ny =30
    nz =30
    nw =30

    lim = np.pi
    
    xs = np.linspace(-lim,lim,nx)
    ys = np.linspace(-lim,lim,ny)
    zs = np.linspace(-lim,lim,nz)
    ws = np.linspace(-lim,lim,nw)
    
    

    

    X, Y, Z, W = np.meshgrid(xs, ys, zs, ws, indexing = 'ij')
    Fs = f(X,Y,Z,W)
    
    
    print("Here")
    Tab1 = vf.InterpTable4D(xs,ys,zs,ws,Fs,kind='cubic',cache = False)
    Tab2 = RegularGridInterpolator((xs,ys,zs,ws),Fs,method='cubic')
    
    FDDerivChecker(Tab1.sf(),[1.,.5,-.6,.3])
    
    
    vf.InterpTable4DSpeedTest(Tab1.sf(),-lim,lim,-lim,lim,-lim,lim,-lim,lim,1000000,False)
    
    
    np.random.seed(1)
    
    samps = (np.random.rand(100,4)-.5)*2*lim
    
    err1,err2 = [],[]
    for samp in samps:
        err1.append(abs(Tab1(*samp) - f(*samp)))
        err2.append(abs(Tab2(samp) - f(*samp)))

    err1 = np.array(err1)
    err2 = np.array(err2)

    print(np.mean(abs(err1)),np.mean(abs(err2)))
    print(abs(err1).max(),abs(err2).max())

    
    

    
    print("Here")

    
    
    
    



