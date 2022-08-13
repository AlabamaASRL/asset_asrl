import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import random as rand
import time

from DerivChecker import FDDerivChecker

import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes


##########################################

g0      =  9.81 
Lstar   =  1000           ## m
Tstar   =  60.0           ## sec
Mstar   =  10             ## kgs
Astar   =  Lstar/Tstar**2
Vstar   =  Lstar/Tstar
Rhostar =  Mstar/Lstar**3
Estar   =  Mstar*(Vstar**2)




CD      = .5
RhoAir  = 1.225     /Rhostar
RhoIron = 7870      /Rhostar
h_scale = 8.44e3    /Lstar
E0      = 400000    /Estar
g       = g0/Astar

###########################################


def MFunc(rad,RhoIron):return (4/3)*(np.pi*RhoIron)*(rad**3)
def SFunc(rad):  return np.pi*(rad**2)
    
    
##############################################################################

class Cannon1(oc.ode_x_u_p.ode):
    def __init__(self, CD,RhoAir,RhoIron,h_scale,g):
        ############################################################
        args  = oc.ODEArguments(4,0,1)
        
        v     = args.XVar(0)
        gamma = args.XVar(1)
        h     = args.XVar(2)
        r     = args.XVar(3)
        
        rad = args.PVar(0)
        
        S    = SFunc(rad)
        M    = MFunc(rad,RhoIron)
        
        rho     = RhoAir * vf.exp(-h / h_scale)
        
        D       = (0.5*CD)*rho*(v**2)*S
        
        vdot     = -D/M - g*vf.sin(gamma)
        gammadot = -g*vf.cos(gamma)/v
        hdot     = v*vf.sin(gamma)
        rdot     = v*vf.cos(gamma)
        
        ode = vf.stack([vdot,gammadot,hdot,rdot])
        ##############################################################
        super().__init__(ode,4,0,1)
        
class Cannon2(oc.ode_x_u_p.ode):
    def __init__(self, CD,RhoAir,RhoIron,h_scale,g):
        ############################################################
        args  = oc.ODEArguments(4,0,1)
        
        v     = args.XVar(0)
        gamma = args.XVar(1)
        h     = args.XVar(2)
        r     = args.XVar(3)
        
        rad = args.PVar(0)
        
        S    = SFunc(rad)
        M    = MFunc(rad,RhoIron)
        
        rho     = RhoAir * vf.exp(-h / h_scale)
        
        D       = (0.5*CD)*rho*(v**2)*S
        
        vdot     = -D/M - g*vf.sin(gamma)
        gammadot = vf.divtest(-g*vf.cos(gamma),v*1)
        hdot     = v*vf.sin(gamma)
        rdot     = v*vf.cos(gamma)
        
        ode = vf.stack([vdot,gammadot,hdot,rdot])
        ##############################################################
        super().__init__(ode,4,0,1)


rad0   = .1 /Lstar
h0     = 100 /Lstar
r0     = 0
m0     = MFunc(rad0,RhoIron)
gamma0 = np.deg2rad(45)
v0     = np.sqrt(2*E0/m0)*.99




IG = np.zeros((6))
IG[0] = v0
IG[1] =gamma0
IG[2] = h0
IG[3] = r0
IG[5] = rad0



v,gamma = Args(2).tolist()

VF = gamma
SF = v

F1 = VF/SF
F2 = vf.divtest(VF,SF)





f1 = Cannon1(CD,RhoAir,RhoIron,h_scale,g).vf()
f2 = Cannon2(CD,RhoAir,RhoIron,h_scale,g).vf()

d1 = F1.computeall([2,3],[1])
d2 = F2.computeall([2,3],[1])

print(v0)

print(d1[0]-d2[0])
print(d1[1]-d2[1])
print(d1[2]-d2[2])

print(d1[3]-d2[3])

print(d1[3])
print(d2[3])




