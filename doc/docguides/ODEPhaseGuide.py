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


phase.addDeltaVarEqualCon(0,1.0)
# These do the same as the following
DeltaEqualCon= Args(2)[1]-Args(2)[0] -1.0
phase.addEqualCon("FirstandLast",DeltaEqualCon,[0])

## These do the same thing, constraining the elapsed time over the phase to be = 1.0
phase.addDeltaVarEqualCon(6,1.0)
phase.addDeltaTimeEqualCon(1.0) #Time is special and has its on named method

# Both are equivalent to the following
DeltaEqualCon= Args(2)[1]-Args(2)[0] -1.0
phase.addEqualCon("FirstandLast",DeltaEqualCon,[6])

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

PhaseRegion = "First"

def AnInequalCon():
    XtU_OP_SP = Args(13)
    return -1.0*XtU_OP_SP

XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
OPVars  = range(0,1)   # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
SPVars  = range(0,2)   # indcices of the phase Static Parameters (indexed from 0) we want to forward to our function

phase.addInequalCon(PhaseRegion,AnInequalCon(),XtUVars,OPVars,SPVars)

# Same rules as covered for addEqualCon
phase.addInequalCon("Path", Args(4).sum(),[0,1,2],[],[1])
phase.addInequalCon("Back",  Args(3).squared_norm()-1,[3,4,5])
phase.addInequalCon("StaticParams",1-Args(2).norm(),[0,1])

###############################################################################

# Add lower bound to the 7th state,time,control variable
PhaseRegion = "Back"
VarIndex    = 7
LowerBound  = 0.0
Scale       = 1.0  # strictly positive scale factor

phase.addLowerVarBound(PhaseRegion,VarIndex,LowerBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addLowerVarBound(PhaseRegion,VarIndex,LowerBound)


# Add upper bound to the 7th state,time,control variable
PhaseRegion = "Back"
VarIndex    = 7
UpperBound  = 1.0
Scale       = 1.0  # strictly positive scale factor

phase.addUpperVarBound(PhaseRegion,VarIndex,UpperBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addUpperVarBound(PhaseRegion,VarIndex,UpperBound)


## Add Both Lower and Upper Bounds at same time
PhaseRegion = "Back"
VarIndex    = 7
LowerBound  = 0.0
UpperBound  = 1.0
Scale       = 1.0  # strictly positive scale factor for both bounds

phase.addLUVarBound(PhaseRegion,VarIndex,LowerBound,UpperBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addLUVarBound(PhaseRegion,VarIndex,LowerBound,UpperBound)

# Also works for the parameter variables
phase.addLUVarBound("StaticParams",0,-1.0,1.0)

# Violations are now of order one
Scale = 10000.0
phase.addUpperVarBound("ODEParams",0,1.0/10000.0, Scale)

##############################################################################


## Upper bound on the norm of the controls
PhaseRegion ="Path"
ScalarFunc = Args(3).norm()
XTUVars = [7,8,9]
UpperBound = 1.0
Scale = 1.0

phase.addUpperFuncBound(PhaseRegion,ScalarFunc,XTUVars,UpperBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addUpperFuncBound(PhaseRegion,ScalarFunc,XTUVars,UpperBound)


## Lower bound on the norm of the controls
PhaseRegion ="Path"
ScalarFunc = Args(3).norm()
XTUVars = [7,8,9]
LowerBound = 0.0
Scale = 1.0

phase.addLowerFuncBound(PhaseRegion,ScalarFunc,XTUVars,LowerBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addLowerFuncBound(PhaseRegion,ScalarFunc,XTUVars,LowerBound)


## Both at the same time
PhaseRegion ="Path"
ScalarFunc = Args(3).norm()
XTUVars = [7,8,9]
LowerBound = 0.0
UpperBound = 1.0

Scale = 1.0

phase.addLUFuncBound(PhaseRegion,ScalarFunc,XTUVars,LowerBound,UpperBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addLUFuncBound(PhaseRegion,ScalarFunc,XTUVars,LowerBound,UpperBound)

##############################################################################


## Upper bound on the norm of the controls
PhaseRegion ="Path"
XTUVars = [7,8,9]
UpperBound = 1.0
Scale = 1.0

phase.addUpperNormBound(PhaseRegion,XTUVars,UpperBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addUpperNormBound(PhaseRegion,XTUVars,UpperBound)


## Lower bound on the norm of the controls
PhaseRegion ="Path"
XTUVars = [7,8,9]
LowerBound = 0.0
Scale = 1.0

phase.addLowerNormBound(PhaseRegion,XTUVars,LowerBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addLowerNormBound(PhaseRegion,XTUVars,LowerBound)


## Both at the same time
PhaseRegion ="Path"
XTUVars = [7,8,9]
LowerBound = 0.0
UpperBound = 1.0

Scale = 1.0

phase.addLUSquaredNormBound(PhaseRegion,XTUVars,LowerBound,UpperBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addLUSquaredNormBound(PhaseRegion,XTUVars,LowerBound,UpperBound)

###########################################################################

VarIdx     = 0
LowerBound = 0.0
Scale      = 1.0
phase.addLowerDeltaVarBound(VarIdx,LowerBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addLowerDeltaVarBound(6,LowerBound)




VarIdx     = 0
UpperBound = 1.0
Scale      = 1.0

phase.addUpperDeltaVarBound(VarIdx,LowerBound,Scale)
# If no scale factor is supplied it is assumed to be = 1.0
phase.addUpperDeltaVarBound(VarIdx,LowerBound)



# Time is special, we can use addLower/UpperDeltaTimeBound instead
LowerBound = .5
UpperBound = 1.5
Scale      = 1.0

phase.addLowerDeltaTimeBound(LowerBound,Scale)
phase.addUpperDeltaTimeBound(UpperBound)

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

def AStateObjective():
    XtU_OP_SP = Args(13)
    return XtU_OP_SP.norm()  ## An Asset Scalar Function

PhaseRegion = "Back"
XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
OPVars  = range(0,1)   # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
SPVars  = range(0,2)   # indcices of the phase Static Parameters (indexed from 0) we want to forward to our function

phase.addStateObjective(PhaseRegion,AStateObjective(),XtUVars,OPVars,SPVars)

##########################################################################




# Minimize the final value of XtUVar 5 
PhaseRegion = "Last"
VarIdx = 5
Scale = 1.0
phase.addValueObjective(PhaseRegion,VarIdx,Scale)


# Maximize the initial value of XtUVar 0 
PhaseRegion = "First"
VarIdx = 0
Scale = -1.0  ## Negative scale factors to maximize!!!
phase.addValueObjective(PhaseRegion,VarIdx,Scale)


# Minimize the Static Param 0 
PhaseRegion = "StaticParams"
VarIdx = 0
Scale = 1.0
phase.addValueObjective(PhaseRegion,VarIdx,Scale)


##############################################################################

# Minimize change in XtUVar 2 across the phase ie: x2_f - x2_0
VarIdx = 2
Scale  = 1.0
phase.addDeltaVarObjective(VarIdx,Scale)


# Maximize change in XtUVar 4 across the phase ie: x4_f - x4_0
VarIdx = 4
Scale  = -100.0  # Negative scale factor to maximize
phase.addDeltaVarObjective(VarIdx,Scale)

# Minimize the duration of the phase : tf-t0
VarIdx = 6  # Index of time
Scale  = 1.0
phase.addDeltaVarObjective(VarIdx,Scale)
## Time is special and has its own named method that does the same as above
phase.addDeltaTimeObjective(Scale)

##############################################################################


########################################################################
########################################################################
########################################################################
########################################################################
########################################################################


def AnIntegrand():
    XtU_OP_SP = Args(13)
    return XtU_OP_SP.norm()  ## An Asset Scalar Function

XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
OPVars  = range(0,1)   # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
SPVars  = range(0,2)   # indcices of the phase Static Parameters (indexed from 0) we want to forward to our function


# Signature if variables of all types are needed by integrand
phase.addIntegralObjective(AnIntegrand(),XtUVars,OPVars,SPVars)

# Signature if only state,time, and control variables needed by integrand
phase.addIntegralObjective(Args(3).norm(),[7,8,9])

# All integrands are minimized, so to maximize, multiply by negative number
phase.addIntegralObjective(-10.0*Args(3).norm(),[7,8,9])



########################################################################
########################################################################
########################################################################
########################################################################
########################################################################


def AnIntegrand():
    XtU_OP_SP = Args(12)
    return XtU_OP_SP.norm()  ## An Asset Scalar Function

XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
OPVars  = range(0,1)   # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
SPVars  = range(0,1)   # indcices of the phase Static Parameters (indexed from 0), MINUS THE ONE WE ARE ASSIGING THE INTEGRAL TOO
IntSPVar = 1 # Assign the value of the intgral to the second static parameter

# Signature if variables of all types are needed by integrand
phase.addIntegralParamFunction(AnIntegrand(),XtUVars,OPVars,SPVars,IntSPVar)

# Signature if only state,time, and control variables needed by integrand
phase.addIntegralParamFunction(Args(3).norm(),[7,8,9],IntSPVar)

## Now we can apply constraints to the integral by constraining the static param
# Ex: constrain the integral to be equal to 100.0
phase.addBoundaryValue("StaticParams",[1],[100.0])


phase.returnStaticParams()

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

def URateBound(LBoundVec,UBoundVec):
        tUtU = Args(8)
        ti,Ui = tUtU.head(4).tolist([(0,1),(1,3)])
        tip1,Uip1 = tUtU.tail(4).tolist([(0,1),(1,3)])
    
        h = tip1-ti
        Urate = (Uip1-Ui)/(h)
    
        UpperBound = Urate - UBoundVec
        LowerBound = LBoundVec - Urate
    
        return vf.stack(UpperBound,LowerBound)

phase.addInequalCon("PairWisePath",URateBound(-np.ones(3),np.ones(3)),[6,7,8,9])
    
###############################################################################

###########################
## Ex. Equality Constraint
edx1 = phase.addBoundaryValue("Front",range(0,5),np.zeros((5)))
edx2 = phase.addEqualCon("Path",Args(3).norm()-1.0,[8,9,10]) 
edx3 = phase.addDeltaVarEqualCon(6,1.0)

## Removal order doesnt matter
phase.removeEqualCon(edx1)
phase.removeEqualCon(edx3)
phase.removeEqualCon(edx2)


#############################
## Ex. Inequality Constraint
idx1 = phase.addInequalCon("Path", Args(4).sum(),[0,1,2],[],[1])
idx2 = phase.addLUVarBound("StaticParams",0,-1.0,1.0)
idx3 = phase.addLUFuncBound("Path",Args(3).norm(),[8,9,10],0,1)

phase.removeInequalCon(idx2)
phase.removeInequalCon(idx3)
phase.removeInequalCon(idx1)

#######################
## Ex. State Objective

sdx1 = phase.addStateObjective("Back",Args(3).squared_norm(),[0,1,2])
sdx2 = phase.addValueObjective("Back",6,1.0)
sdx3 = phase.addDeltaTimeObjective(1.0)

phase.removeStateObjective(sdx3)
phase.removeStateObjective(sdx2)
phase.removeStateObjective(sdx1)


##########################
## Ex. Integral Objective
intdx1 = phase.addIntegralObjective(Args(3).norm(),[7,8,9])

phase.removeIntegralObjective(intdx1)

##########################
## Ex. Integral Parameter Function
ipdx1 = phase.addIntegralParamFunction(Args(3).norm(),[7,8,9],IntSPVar)

phase.removeIntegralParamFunction(ipdx1)

###############################################################################

edx = phase.addBoundaryValue("Front",range(0,5),np.zeros((5)))
idx = phase.addLUVarBound("Path",0,-1.0,1.0)

phase.optimize()

ecvals = phase.returnEqualConVals(edx)
ecmults = phase.returnEqualConLmults(edx)

print(ecvals[0])
print(ecmults[0])


icvals = phase.returnInequalConVals(idx)
icmults = phase.returnInequalConLmults(idx)

for icval,icmult in zip(icvals,icmults):
    print(icval)
    print(icmult)
    




    


