import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import random as rand
import time

from DerivChecker import FDDerivChecker

vf    = ast.VectorFunctions
Args = vf.Arguments


def Eanom():
    
    E,e,M = Args(3).tolist()
    
    fE   = E - e*vf.sin(E) - M
    dFE  = 1 - e*vf.cos(E)
    d2FE = e*vf.sin(E)
    return vf.ScalarRootFinder(fE,dFE,10,1.0e-12)



Efunc = Eanom()
Kfunc = ast.Astro.Kepler.KeplerPropagator(1.0).vf()

M = 1.0
e = .01

E = Efunc([M,e,M])

Efunc.SpeedTest([M,e,M],1000000)
Kfunc.SpeedTest([1,0,0.1,0,1,0,1],1000000)
ast.Astro.CartesianToClassic(1.0).vf().SpeedTest([1,0,.01,0,1,.1],100000)


print(Efunc.jacobian([M,e,M]))

print(Efunc.adjointhessian([M,e,M],[1]))

FDDerivChecker(Efunc,[M,e,M])








    
    