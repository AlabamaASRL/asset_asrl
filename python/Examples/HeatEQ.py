import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import seaborn as sns

norm      = np.linalg.norm
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
solvs     = ast.Solvers



class HeatEq(oc.ode_x_u.ode):
    
    def __init__(self,):
        
        args = oc.ODEArguments(4,1)
        
        super().__init__(ode,4,1)
        

