import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes

'''
Hyper-Sensitive Problem
https://openmdao.github.io/dymos/examples/hypersensitive/hypersensitive.html

'''

class HyperSens(oc.ode_x_u.ode):
    def __init__(self):
        ############################################################
        args  = oc.ODEArguments(2,1)
        
        x0 = args.XVar(0)
        u   =args.UVar(0)
        
        x0dot = -x0 + u
        jdot  = (u**2 +x0**2)/2.0
        
        ode = vf.stack(x0dot,jdot)
        ##############################################################
        super().__init__(ode,2,1)

