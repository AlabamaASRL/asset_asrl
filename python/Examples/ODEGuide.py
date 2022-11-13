import asset_asrl as ast
import numpy as np


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments

############### Intro #####################
'''
One of ASSET's primary goals is facilitating optimal control and intergation of dynamical systems governed by ordinary differential equations (ODEs).
In ASSET, an ODE is simply a vector function adhering to the state-space formalism shown below. 
That is, it is a vectorfunction that takes as arguments the state of the system, x, the current time,
t, time varying controls, u, some static ODE parameters p, and returns the time derivative of the state variables. 

Some ODEs do not require the u or p inputs (e.g. ballistic gravity model), and thus it is not necessary to provide them in those cases. 
However, because we use a common interface between autonomous and non-autonomous ODEs, it is always necessary to provide 
an explicit time variable even if it is unused in the dynamics. 
'''

'''
ODE's can be written using any of the techniques described in the VectorFunctions
section, provided that they obey the state space formalism for their inputs and outputs. 
For example a simple two body gravitational model with low thrust could be written as shown 
below. This model would have 6 state variables representing the position an dvelocity vectors
relative to the central body, and three control variables reperesenting the thrust direction and engine throttle.
We implictly assume that we will later place an upper bound of 1 on the norm of the control so that the accelleration
never exceeds the maximum we specify here. By convention, the psotion and veclity state varainles will be
the first 6 input arguments, followed by the time, and then the controls. As outputs we return
the time derivatives of position and velocity in the same order we assumed in the input arguments.
'''

def TwoBodyLTFunc(mu,MaxLTAcc):
    
    XtU = Args(10) # [ [R,V] , t, [U]]
    
    R,V,t,U  = XtU.XVec().tolist([(0,3),(3,3),(6,1),(7,3)])
    
    G = -mu*R.normalized_power3()
    
    Acc = G + U*MaxLTAcc
    
    Rdot = V
    Vdot = Acc
    
    ode = vf.stack([Rdot,Vdot])
    return ode

    

########### ODEArguments ###############

'''
To simplify the process of defining Vector functions adhering to the stae space
formalism, we provde the ODEArguments class inside of the optimal control module.
This class is a thin wrapper around the Arguments class that allows you to index relevant
subvectors and elements of an ODE's inputs in clearer way than using Arguments. To construct 
ODEArguments we pass the number of state variables,control varibles, and ODE parameters (if)
any. The total input size will be the sum of XVars,PVars, and UVars plus 1 for time. We can the
address the relevant subvectors of our input using the X/U/PVec() methods. These methods are return regular
segment types so we can then apply all operations we would to those objects. Similarly, we can also adress
specific elemnts of each of these subvectors using the X/U/PVar(i) methods.
'''



def TwoBodyLTFunc(mu,MaxLTAcc):
    
    XVars = 6
    UVars = 3
    PVars = 0
    
    XtU = oc.ODEArguments(XVars,UVars,PVars) # [ [R,V] , t, [U],[]]
    
    # no need to specify Pvars if there arnet any, same would go for Uvars
    # if there were no control variables
    XtU = oc.ODEArguments(XVars,UVars) 
        
    
    # Index state,control or parameter vectors
    
    XVec = XtU.head(6)
    XVec = XtU.XVec()
    
    UVec = XtU.segment(XVars+1,Uvars)
    UVec = XtU.UVec()
    
    r0,r1,r2,v0,v1,v2 = XVec.tolist()
    
    u0,u1,u2 = UVec.tolist()
    
    # If we had ode parameters
    #PVec = XtU.segment(XVars+1+UVars,PVars)
    #PVec = XtU.PVec()
    
    
    R,V  = XVec.tolist([(0,3),(3,3)])
    
    U = UVec
    
    ### Index specific elements 
    t  = XtU.TVar()   # same as XtU[Xvars]
    
    v1 = V[1]
    v1 = XVec[4]
    v1 = XtU.XVar(4) # XtU.UVar(i) is same as XtU[i]
    
    u0 = UVec[0]
    u0 = XtU.UVar(0) # XtU.UVar(i) is same as XtU[Xvars+i]
    
    
    G = -mu*R.normalized_power3()
    Acc = G + U*MaxLTAcc
    
    Rdot = V
    Vdot = Acc
    
    ode = vf.stack([Rdot,Vdot])
    return ode

########### Defining An ODE Classes ###############

'''
If you were to inspect the type of the result of the function above, it would
be VectorFunction, and at this point ASSET has no idea that it is an ODE.
For ASSET to recongnize our function as an ODE and allow ua to use it directly with
all of associated utilities, we need to define it using the class
based style describing in VectorFunctionGuide, but Inherit from the class oc.ODEBase
rather vector function. Therefore, the correct way to write the TwoBody ODE shown below.
When initializing our base class we simply supply, the asset vector function specifying
the ode as well as the number of states, controls, and parameters.
'''


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
        


        
ode = TwoBodyLTODE(1,.01)

phase = ode.phase("LGL3")
integ = ode.integrator("DOPRI87",.1)

help(integ)

'''
This object is now a full fledged ode, from which we can dispatch phase and integrator
objects. We will dicuss usage of these in the next sections.
'''



