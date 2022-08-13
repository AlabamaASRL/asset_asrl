import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import random as rand
import time

from DerivChecker import FDDerivChecker

vf    = ast.VectorFunctions
Args = vf.Arguments



def C2(tol = 1.0e-10):
    psi = Args(1)[0]
    sqsi = vf.sqrt(psi)
    
    return vf.ifelse(psi>tol, 
              (1.0 - vf.cos(sqsi)) / psi,
              (1.0 - vf.cosh(vf.sqrt(-psi))) / psi)
def C3(tol = 1.0e-10):
    psi = Args(1)[0]
    sqsi = vf.sqrt(psi)
    
    return vf.ifelse(psi>tol, 
              (sqsi - vf.sin(sqsi)) / (sqsi * psi),
              (vf.sinh(vf.sqrt(-psi)) - vf.sqrt(-psi)) / vf.sqrt(-psi * psi * psi))
        
    
def LUniv(mu):
    
    psi,rsum,A = Args(5).tolist()
    
    c2 = C2().eval(psi)
    c3 = C3().eval(psi)
    
    
    

    
