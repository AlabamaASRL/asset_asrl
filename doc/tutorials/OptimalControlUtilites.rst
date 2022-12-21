#########################
Optimal Control Utilities
#########################


LGLInterpTable and InterpFunction
#################################
The LGLInterpTable class inside of the optimal control module facilitates the interpolation of time-series data expressed in the
ODE format. This object is used to facilitate the reintegration of converged trajectories or to incorporate
arbitrary time-series data into a vector function expression. It is distinct from InterpTable1D discussed in the vector functions
section.


.. code-block:: python
	
	import asset_asrl as ast
    import matplotlib.pyplot as plt
    import numpy as np


    vf = ast.VectorFunctions
    Args = vf.Arguments
    oc = ast.OptimalControl

    class TwoBodyLTODE(oc.ODEBase):
    
        def __init__(self,mu,MaxLTAcc):
            XVars = 6
            UVars = 3
       
            XtU = oc.ODEArguments(XVars,UVars)
        
            R,V  = XtU.XVec().tolist([(0,3),(3,3)])
            U = XtU.UVec()
        
            G = -mu*R.normalized_power3()
            Acc = G + U*MaxLTAcc
        
            Rdot = V
            Vdot = Acc
            ode = vf.stack([Rdot,Vdot])
            super().__init__(ode,XVars,UVars)


    def ULaw(throttle):
        V = Args(3)
        return V.normalized()*throttle

	

