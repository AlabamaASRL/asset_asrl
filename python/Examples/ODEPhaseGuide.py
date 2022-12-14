import asset_asrl as ast
import numpy as np
import matplotlib.pyplot as plt


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
ODEArgs   = oc.ODEArguments


############# Intro ########################
'''
In this section, we will explain the usage of ASSETs phase object. Blah Blah
'''


'''
As an introduction, we will first walk through the phase API for the constrived ODE shown
below, which has both controls and ODE parameters. At the end we will apply what we have
learned to solve a realistic optimal control problem.
'''

class DummyODE(oc.ODEBase):
    def __init__(self,xv,uv,pv):
        args = oc.ODEArguments(xv,uv,pv)
        super().__init__(args.XVec(),xv,uv,pv)
        

t0 = 0
tf = 0
        
XtUP0 = np.ones(11)
XtUP0[6] = t0

XtUPf = np.ones(11)
XtUPf[6] = tf

InitialGuess = [XtUP0,XtUPf]

ode = DummyODE(6,3,1)


####################
#Initialixzation
'''
With this initial guess in hand, we can now go about constucting the phase object
using the .phase method of the ode. At minimum, we must first specify the transcription
mode for the phase dynamics as as string. Here we have chosen, the  third order Legendres gauss
lobatto collocation or LGL3, which approximates the trajectory as piecewise cubic splines. We can also
choose from the 5th and 7th Order LGL methods, the trapezoidal method, or a central shooting scheme. In most
cases we suggest first trying the LGL3 scheme, however 5th and 7th methods may be superior for some applications.
Additionally, users should prefer the LGL collocation methods over the central shooting scheme
for almost all applications, as they are almost always significantly faster and more robust.
'''

######################

#Options: Trapezoidal,LGL3,LGL5,LGL7, CentralShooting
phase = ode.phase("LGL3")

'''
We can now initialize out phase an Initial guess for the trajectory
using the setTraj method. In most cases we will just pass in the initial guess
and specify the number of segments of the chosen transcription type we want to
use to approximate the dynamics. By default these will be evenly spaced in time.
Note that the number of segments does not have to match the number
of states in the initial guess, nor do the states in the inital guess have to be evenly spaced.
We can also manually specify the initial spacing for segments. This is done by passing a python list
SegBinSpacing of lenght n>=2 specifying the spacing on the non-dimensional time interval 0-1
for groups of linearly spaced segments. We then pass another list of lenght n-1 specifying
the number of segments we want in each group. For example, we can replicate the bahavior of the
default method as shown below. Or alternatively, we could specify that we want to vary the density of 
segments across the phase. In most cases, users first option should be to jsut evenly space segments over
the phase. One can also intialize the transcription method and initial guess in the same
call as shown on the final line.
'''

## 500 Segments evenly spaced over entire time interval
phase.setTraj(InitialGuess,500)

## 500 Segments evenly spaced over entire time interval
SegBinSpacing = [0.0 , 1.0]
SegsPerBin=        [500]
phase.setTraj(InitialGuess,SegBinSpacing,SegsPerBin)

# 300 segments spaced over first half of trajectory, 200 over last half
SegBinSpacing = [0.0 , 0.5 , 1.0]
SegsPerBin       =[ 300 , 200]
phase.setTraj(InitialGuess,SegBinSpacing,SegsPerBin)

## Set Transcription, IG, and number of segments
phase = ode.phase("LGL3",InitialGuess,500)



phase.setStaticParams([0.0,0.0])


#####################################

PhaseRegion = "First"

def AnEqualCon():
    XtU_OP_SP = Args(13)
    return XtU_OP_SP

XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
OPVars  = range(0,1)   # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
SPVars  = range(0,2)   # indcices of the phase Static Parameters (indexed from 0) we want to forward to our function

phase.addEqualCon(PhaseRegion,AnEqualCon(),XtUVars,OPVars,SPVars)

#################################################################

PhaseRegion = "Last"

## Only needs thrid and second state varibales, the first ode parameter, and the second static parameter
def AnotherEqualCon():
    x1,x2,op0,sp1 = Args(4).tolist()
    return vf.sum(x1,x2,op0/sp1) + 42.0

XtUVars = [1,2]  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
OPVars  = [0]    # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
SPVars  = [1]    # indcices of the phase Static Parameters (indexed from 0) we want to forward to our function

phase.addEqualCon(PhaseRegion,AnotherEqualCon(),XtUVars,OPVars,SPVars)


##################################################################

XtUVars = [7,8,9]  # Just the controls and nothing else
# enforce unit norm of all control vectors
phase.addEqualCon("Path",Args(3).norm()-1.0,XtUVars,[],[]) 
# same as above
phase.addEqualCon("Path",Args(3).norm()-1.0,XtUVars) 


OPVars = [0]  # Just the ODEParam
#Enforce Square of first ODE param = 4
phase.addEqualCon("ODEParams",Args(1)[0]**2 - 4.0,[],OPVars,[]) 
# same as above
phase.addEqualCon("ODEParams",Args(1)[0]**2 - 4.0,OPVars) 


SPVars = [0,1]  # Just the static params
#Enforce sum of static params = 2
phase.addEqualCon("StaticParams",Args(2).sum() - 2.0,[],[],SPVars) 
# same as above
phase.addEqualCon("StaticParams",Args(2).sum() - 2.0,SPVars) 

################################################################

def FrontBackEqCon():
    X_0,t_0,X_f,t_f,sp0 = Args(15).tolist([(0,6),(6,1),(7,6),(13,1),(14,1)])
    
    eq1 = X_0-X_f
    eq2 = t_f-t_0 - sp0
    return vf.stack(eq1,eq2)
     

XtUVars = range(0,7)  # index of time
SPVars  = [0]  # first static parameter
# Constrain first and last states to be equal and
# constrain Delta Time over the phase (tf-t0) to be equal to the first static parameter
phase.addEqualCon("FirstandLast",FrontBackEqCon(),XtUVars,[],SPVars)

##################################################################

XtUVars = [1,3,9]
Values  = [np.pi,np.e,42.0]
phase.addBoundaryValue("First",XtUVars,Values)

OPVars = [0]
Values  = [10.034]
phase.addBoundaryValue("ODEParams",OPVars,Values)

SPVars = [0,1]
Values  = [1.0,4.0]
phase.addBoundaryValue("StaticParams",SPVars,Values)

