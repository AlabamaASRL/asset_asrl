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
As introduction, we will walk through using phase to optimize a low-thrust
transfer between two-circular orbits using our TwoBodyLTODE model. In particular,
we will attempt to compute a transfer from a fixed circular orbit at radius 1,
to a circular orbit at r = 2.
'''

class TwoBodyLTODE(oc.ODEBase):
    
    def __init__(self,mu,MaxLTAcc):
        
        XVars = 6
        UVars = 3
        
        
        XtU = oc.ODEArguments(XVars,UVars)
        
        R,V  = XtU.XVec().tolist([(0,3),(3,3)])
        U = XtU.UVec()
        
        G = -mu*R.normalized_power3()
        Acc = G + (U*MaxLTAcc)
        
        Rdot = V
        Vdot = Acc
        
        ode = vf.stack([Rdot,Vdot])
        
        super().__init__(ode,XVars,UVars)

'''
As a first step, we will generate an initial guess for our transfer using the
ODEs integrator with a prograde control law and a terminal event.
'''

def ULaw(throttle):
    V = Args(3)
    return V.normalized()*throttle
def RStop(rmax):
    X = Args(10)
    return X.head3().norm()-rmax

ode = TwoBodyLTODE(1,.015)
integULaw   = ode.integrator("DOPRI87",.1,ULaw(0.8),[3,4,5])


r0  = 1.0
v0  = 1.0
t0 = 0.0
tf = 10000.0


X0t0U0 = np.zeros((10))
X0t0U0[0]=r0
X0t0U0[4]=v0
X0t0U0[6]=t0        

TrajULaw,Events   = integULaw.integrate_dense(X0t0U0,tf,[(RStop(2),0,1),])

TT = np.array(TrajULaw).T
plt.plot(TT[0],TT[1],label='Control Law Initial Guess')


plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.axis("Equal")
plt.grid(True)
        
plt.show()

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
phase.setTraj(TrajULaw,500)

## 500 Segments evenly spaced over entire time interval
SegBinSpacing = [0.0 , 1.0]
SegsPerBin=        [500]
phase.setTraj(TrajULaw,SegBinSpacing,SegsPerBin)

# 300 segments spaced over first half of trajectory, 200 over last half
SegBinSpacing = [0.0 , 0.5 , 1.0]
SegsPerBin       =[ 300 , 200]
phase.setTraj(TrajULaw,SegBinSpacing,SegsPerBin)

## Set Transcription, IG, and number of segments
phase = ode.phase("LGL3",TrajULaw,500)

######################################
'''
In addition to specifying the trnacription mode, we can also choose from several
different control regularizations, using the .setControlMode method. By default,
this is set to FirstOrderSpline, which will ensure that the control hsitory has smooth
first deriavtives (if possible for the chosen transcription). This typcally sufficient to prevent control chattering in
the LGL5, and LGL7 methods. We can also set the ControlMode to HighestOrderSpline to enforece continuity
in all deriavtives possible for a given transcription method. For LGL5, the control is represented as piecewise
quadratic function, so FirstOrderSPline and HighestOrderSpline are equavalent. For LGL7, the control is represented
a piecwise cubic function, therfore setting control mode to Higest order SPline will ensures that this cubic function
has smooth first and second derivatives. For the LGL3,Trapezoidal,and Central Shooting schemes, the control
history is piecewise linear across segments and does need any regularization, thus for those methods, FirstOrderSpline and 
HighestOrderSpline have no effect.
  

Alternatively, For all methods, we can also specify that rather than having smooth control history, we want to have a piecewise
constant control history with 1 uniquye control per segment. This can be specified by setting the control mode to BlockConstant.
In our experience this control parameterization can be very robust and typically results in KKT matrices that are faster to factor.
The caveat is that special care must be taken when reintegrating converged solutions with an explicit integrator. This will be covered in a later section. 
'''

# Options: FirstOrderSpline,HighestOrderSpline,BlockConstant,NoSpline
phase.setControlMode("FirstOrderSpline")
##############################################################
##############################################################
# Constraints and Objectives
'''
We will now cover how one applies constraints and objectives to a phase in order to
define an optimization problem. Througout these sections we will first show the most
general way of achieving a partiular task, as well as more
specific specialized methods that accomplish the same thinf. Users are always recommended to use
the specialized methods when applicable, but can always fall back on the general form if what
they are trying to accomplish is not possible with the simplified interface.
'''

'''
In asset, there are 5 different general types of constraints/objectivs that
can be applied to a phase. EqualityConstraints, InequalityContraints, StateObjectives,
IntegralObjectives, and Integral Parameter relations. Equality constraints of functions
of the phase variables of the form f([X_i,t_i,U_i])=0
'''

#############################################################
# Equality constraints


phase.addBoundaryValue("Front",range(0,7) , X0t0U0[0:7])

##############################################################
phase = ode.phase("LGL3",TrajULaw,300)
phase.setControlMode("BlockConstant")

phase.addBoundaryValue("Front",range(0,7) , X0t0U0[0:7])
phase.addLUNormBound("Path"   ,range(7,10),.0001,1.0)
phase.addLowerNormBound("Path",range(0,3) ,.99999)


def CircularOrbit(r):
    R,V = Args(6).tolist([(0,3),(3,3)])
    eq1 = R.norm()-r
    eq2 = V.norm() - np.sqrt(1/r)
    eq3 = V.dot(R)
    return vf.stack(eq1,eq2,eq3)

phase.addEqualCon("Back",CircularOrbit(2.0),range(0,6),[],[])

#phase.addDeltaTimeObjective(1)
phase.addIntegralObjective(Args(3).norm()/10,range(7,10))
phase.addUpperDeltaTimeBound(TrajULaw[-1][6]*1.00,.1)
phase.optimizer.set_OptLSMode("L1")
phase.optimizer.MaxLSIters=2
#phase.solve_optimize()






#################################################

TT = np.array(TrajULaw).T
plt.plot(TT[0],TT[1],label='80% Prograde Throttle')

TT = np.array(phase.returnTraj()).T
plt.plot(TT[0],TT[1],label='Solved')


plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.axis("Equal")
plt.grid(True)
        
plt.show()
TT = np.array(phase.returnTraj()).T
plt.plot(TT[6],TT[7],label='Solved')
plt.plot(TT[6],TT[8],label='Solved')
plt.plot(TT[6],TT[9],label='Solved')
plt.show()


TT = np.array(phase.returnTraj()).T
plt.plot(TT[6],TT[0],label='Solved')
plt.plot(TT[6],TT[1],label='Solved')
plt.show()

############# Transcriptions ##############


#############  ##############




