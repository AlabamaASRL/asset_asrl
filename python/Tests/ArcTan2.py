import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import random as rand
import time

from DerivChecker import FDDerivChecker

vf    = ast.VectorFunctions
Args = vf.Arguments

def Model():
    RV = Args(9)
    # p,v,acc
    R = RV.head3()
    V = RV.segment3(3)
    A = RV.tail3()
    f = vf.stack(V, - R.normalized_power3() + A)
    return func
    
def Thruster():
    RU = Args(6)    
    thrust = RU.tail3()/RU.head3().squared_norm()
    return thrust
    
def Thruster2(R,U):
    thrust = R/U.squared_norm()
    return thrust
    
def Dynamics():
    
    RVU = Args(9)
    R = RVU.head3()
    V = RVU.segment3(3)
    U = RVU.tail3()
    
    ThrusterArgs = vf.stack(R,U)
    
    thrust = Thruster().eval(ThrusterArgs) 
    thrust = Thruster2(R,U)
    
    ModelArgs = vf.stack(R,V,thrust)
        
    ode = Model().eval(ModelArgs)
    
    return ode
    

    




if __name__ == "__main__":
    
    
    X,Y,Z = Args(3).tolist()
    
    F1 = vf.arctanh(X**2)
    F2 = ast.Astro.J2Cartesian(1, .01, 1)
    
    XX = [1,1,1,0,0,1]
    
    FDDerivChecker(F2,XX)

    

    
    

    
    
    
    
    
    
   