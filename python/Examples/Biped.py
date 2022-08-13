import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from DerivChecker import FDDerivChecker

vf = ast.VectorFunctions
oc = ast.OptimalControl
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Args = vf.Arguments


mvec = np.array([3.2,6.8,20,3.2,6.8])
Ivec = np.array([.93,1.08,2.22,.93,1.08])
Lvec = np.array([.4,.4,.625,.4,.4])
dvec = np.array([.128,.163,.2,1.128,.163])


def PlotConfig()



class CartPole(oc.ode_x_u.ode):
    
    def __init__(self,mvec,Ivec,Lvec,dvec):
        
        
        args = oc.ODEArguments(10,5)
        q  = args.head(5)
        qd = args.segment(5,5)
        u  = args.tail(5)
        
        def ni(i):
            if(i==2 or i==4):
                return 0
            else:
                return sum(mvec[i+1:5])*Lvec[i]
                
        def pij(i,j):
            
            if(i==j):
                Ivec[i] + mvec[i]*dvec[i]**2 + ni(i)*Lvec[i]
            elif(i==2 and j>i):
                return 0.0
            else:
                return mvec[j]*dvec[j]*Lvec[i] + ni(j)*Lvec[i]
        
        def qij(i,j):
            if((j<3 and i<3) or (j>2 and i>2)):
                return q[i] -q[j]
            else:
                return q[i] + q[j]
        
        
        
        
        
        
       # super().__init__(ode,10,5)