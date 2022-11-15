import asset_asrl as ast
import numpy as np
import matplotlib.pyplot as plt


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments


'''
In this section, we will describe usagage of asset's integrator object. To 
start we will define an ODE which we want to integrate using what we have learned from the
previous two sections. As concrete example we use a ballistic two body
model whose state varaibles are the position and velocity relative to a central
with gravitationl parameter is mu. This ODE may be defined as shown below. 
For this example,we will be working in canonical units where mu=1.
'''

class TwoBody(oc.ODEBase):
    
    def __init__(self,mu):
        
        XVars = 6
        XtU = oc.ODEArguments(XVars)
        
        R,V  = XtU.XVec().tolist([(0,3),(3,3)])
        
        G = -mu*R.normalized_power3()
        
        Rdot = V
        Vdot = G

        ode = vf.stack([Rdot,Vdot])
        super().__init__(ode,XVars)
        

TBode = TwoBody(1.0)

######### Initilization ##############

'''

To start, we can create an integrator for our ODE by using the .integrator method.
At initialzation we can spicfy the integration scheme as a string as well as the
default stepsize to be used. At the moment we provide the support apaditve dormand prince
integration schemes of order 8(7) and 5(4). These are specified by the strings DOPRI87 or
DOPR54. If no string is specified, the integration we will default to "DOPRI87". The second
argument, DefStepSize is first trial stepsize in adaptive integration algorithm, which will then
adjust the stepsize up or down to meet specifed error tolerances. By default, we thens set the
integrators minimum and maximum allowable stepsizes to be 10000 times smaller and larger than
the default stepsize. However, you can overide this by manually setting all stepsizes using the
.setStepSizes method.
'''

TBode = TwoBody(1.0)

DefStepSize = .1

#Initialization
TBInteg = TBode.integrator("DOPRI87",DefStepSize)
TBInteg = TBode.integrator("DOPRI54",DefStepSize)
TBInteg = TBode.integrator(DefStepSize)  # Default is DOPRI87


print("Default StepSize    = DefStepSize = "      , TBInteg.DefStepSize) 
print("Default MinStepSize = DefStepSize/10000 = ", TBInteg.MinStepSize) 
print("Default MaxStepSize = DefStepSize*10000 = ", TBInteg.MaxStepSize) 

MinStepSize = 1e-6
MaxStepSize = 1.0

## Set def,min, and max step sizes manually
TBInteg.setStepSizes(DefStepSize,MinStepSize,MaxStepSize)

'''
Both integration schemes use the standard absolute and relative tolerance
metrics to asses the accuracy of steps and updadte the stepsize adaptively.
By default we set the absolute tolerance on all state varaibles equal to 1.0e-12
and the relative tolerance to 0.0. This usually works well for well scaled dynamics 
eqations such as our two-body model with mu = 1. However, you can set them manually
yourself using the setAbs/RelTol methods as shown below.

'''

TBode = TwoBody(1.0)
DefStepSize = .1
TBInteg = TBode.integrator(DefStepSize)  # Default is DOPRI87

print("Default AbsTols = [1.0-12...] = ",TBInteg.getAbsTols())
print("Default RelTols = [1.0-12...] = ",TBInteg.getRelTols())

AbsTol      = 1.0e-13
RelTol      = 0

# Set tolerances uniformly for all state variables
TBInteg.setAbsTol(AbsTol)
TBInteg.setRelTol(RelTol)

AbsTols = np.array([1,1,1,3,3,3])*1.0e-13
RelTols = np.array([0,0,0,1,1,1])*1.0e-9
# Set tolerances individually for each state variables
TBInteg.setAbsTols(AbsTols)
TBInteg.setRelTols(RelTols)



######### Standard Integration Methods ##############

'''
 Now that we have convered initalizing integrators, lets examine how we actally
 use them. By far the most used methods are integrate and integrate_dense. Bouth methods,
 take as the first input a full-state vector containg the initial state, time, controls, and
 parameters as well as the final time that we wish to integrate these initial inputs to.
 
The .integrate() method  integrates this initial full-state input vector to final time tf and returns just the full-state at the final time.
integrate_dense takes the same inputs but returns all intermedaite full-states 
calculated by the integrator as single python list. We also call privide integrate_dense 
with an additional arguments sepcifying that we would like to return n evenely spaced steps
between t0 and tf rather than the exact steps taken by the solver. These n states and controls
will be calculated from the exact steps taken by the integrator using a fifth order interpolation method. 
For the DOPRI54 method, interpolated states have efectively
the exact same error as the true steps taken by the integrator. However, for the DOPRI87 method, interpolated states
will have the larger locals error owing the difference in order between the integration and interpolation. In practice
the maximum local error at any point along the trajectory is typcally 2 orders of magnite larger than the integration tolerances. 
 
'''

TBode = TwoBody(1.0)
DefStepSize = .1
TBInteg = TBode.integrator(DefStepSize)  


r  = 1.0
v  = 1.1
t0 = 0.0
tf = 10.0
N  = 100

X0t0 = np.zeros((7))
X0t0[0]=r
X0t0[4]=v
X0t0[6]=t0

# Just the final full-state
Xftf = TBInteg.integrate(X0t0,tf)


TrajExact  = TBInteg.integrate_dense(X0t0,tf)
TrajInterpN = TBInteg.integrate_dense(X0t0,tf,N)


fig,axs = plt.subplots(1,2)

TT = np.array(TrajExact).T
axs[0].plot(TT[0],TT[1],label='TrajExact',marker='o')

TT = np.array(TrajInterpN).T
axs[1].plot(TT[0],TT[1],label='TrajInterpN',marker='o')

for ax in axs:

    ax.scatter(X0t0[0],X0t0[1],color='g',zorder=10,label='X0t0')
    ax.scatter(Xftf[0],Xftf[1],color='r',zorder=10,label='Xftf')
    ax.axis("Equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
axs[0].legend()
axs[1].legend()


plt.show()

'''

'''

######### Event Detection  ##############
'''
We can also pass a list of events to be deteted during the integration. An single 
event is defined as a tuple consisting of: An asset scalar function whose zeros
determine the locations of the event, a direction indicating whether we want to track ascending,descending or all zeros, and A stop code
signifying whether intergation stop after encountering a zero. The scalar function should take the same arguments as the underlying ODE.
The direction flag should be set to 0 to capture all zeros,-1 to capture only zeros where the function value is decresing, or 1 to copatute
zeros where it is increasing. The stopcode should be set to 0 or False if you do not want an event to stop integration. To stop after 1 occurrance,
stopcode can be set to 1 or True. The stopcode can also be set to any positve integrer, in which case it specifes that the number of zeros to be encountered
before stoping. When events are appended to an integration call, in addition to the normal return value, a list of lists of the exact full-states whre each event occurred is
also returned. As an example, the code below will calculate the apopses and periapses of an orbit, and stop after both have been found. Exant roots
of events are found using a newton raphson mehtod appplied to the fifth order spline contnious rperesentation of the trjaectory. The root tolerance
and maximum newton interations may be spcified by modifying the EventTol and MaxEvent Iters firlds of the integrator. Thes default, to 10 and 1e-6 respecively.


'''


#################
r  = 1.0
v  = 1.1
t0 = 0.0
tf = 100.0
N  = 1000


X0t0 = np.zeros((7))
X0t0[0]=r
X0t0[4]=v
X0t0[6]=t0

def ApseFunc():
    R,V = Args(7).tolist([(0,3),(3,3)])
    return R.dot(V)

direction = -1
stopcode = False
ApoApseEvent  = (ApseFunc(),direction,stopcode)


direction = 1
stopcode = False
PeriApseEvent  = (ApseFunc(),direction,stopcode)


direction = 0
stopcode  = 2  # Stop after finding 2 apses
AllApseEvent  = (ApseFunc(),direction,stopcode)


Events = [ApoApseEvent,PeriApseEvent,AllApseEvent]


TBInteg.EventTol =1.0e-10
TBInteg.MaxEventIters =12

## Just append event list to any normal call
Xftf, EventLocs = TBInteg.integrate(X0t0,tf,Events)

Traj, EventLocs  = TBInteg.integrate_dense(X0t0,tf,Events)

Traj, EventLocs  = TBInteg.integrate_dense(X0t0,tf,N,Events)

#EventLocs[i] will be empty if the event was not detected

ApoApseEventLocs  = EventLocs[0]
ApoApse =ApoApseEventLocs[0]

PeriApseEventLocs = EventLocs[1]
PeriApse =PeriApseEventLocs[0]

# Or
AllApseEventLocs  = EventLocs[2]
ApoApse  = AllApseEventLocs[0]
PeriApse = AllApseEventLocs[1]


Traj, EventLocs  = TBInteg.integrate_dense(X0t0,tf,Events)


fig,axs = plt.subplots(1,2)

TT = np.array(Traj).T
EventVals = [ApseFunc()(T)[0] for T in Traj]
axs[1].plot(TT[6],EventVals,marker='o',label='Traj')
axs[1].grid(True)
axs[1].set_ylabel(r"$\vec{R}\cdot\vec{V}$")
axs[1].set_xlabel(r"$t$")

axs[1].scatter(ApoApse[6],0,color='r',zorder=10,label='Apoapse',marker='*',s=100)
axs[1].scatter(PeriApse[6],0,color='b',zorder=10,label='Periapse',marker='*',s=100)


axs[0].plot(TT[0],TT[1],marker='o',label='Traj')

axs[0].scatter(ApoApse[0],ApoApse[1],color='r',zorder=10,label='Apoapse',marker='*',s=100)
axs[0].scatter(PeriApse[0],PeriApse[1],color='b',zorder=10,label='Periapse',marker='*',s=100)
axs[0].axis("Equal")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].legend()
axs[0].grid(True)
plt.show()

##############  Derivatives  ##############

'''
In asset integrators themselves are vector functions, and have analytic first and secodn
deriavtives. The input arguments for the inegrator when used as a vector function conists of
the full-state to be integrated and the final time tf, and the output is the full-state at time tf.
For example, calling compute as shoen bwloe is equavalent to the norma integrate call. This also means
that we can calculate the jacobain and adoint hessian as well.
'''
r  = 1.0
v  = 1.1
t0 = 0.0
tf = 20.0


X0t0 = np.zeros((7))
X0t0[0]=r
X0t0[4]=v
X0t0[6]=t0

X0t0tf = np.zeros((8))
X0t0tf[0:7]=X0t0
X0t0tf[7]=tf



Xftf = TBInteg.integrate(X0t0,tf)

# Same as above
Xftf = TBInteg.compute(X0t0tf)

Jac =  TBInteg.jacobian(X0t0tf)
Hess = TBInteg.adjointhessian(X0t0tf,np.ones((7)))

'''
We should note that the jacobian of an intergator is the same as the state transition matrix (STM).
Since calculation of an ODE's state transition matrix (STM), is critical to the assessing
the stability of periodic orbits, we also provide methods to calculate the STM through the integrator
interface using the integrate_stm methods, which can be used as shown below.
'''

Xftf,Jac = TBInteg.integrate_stm(X0t0,tf)

## With Events

Xftf,Jac, EventLocs = TBInteg.integrate_stm(X0t0,tf,Events)

######### Parrellel Methods  ##############

'''
Finally, for all previusly discsed .iintegrate methods, we have corresponding multithreaded _parallel
verison which will integrate lists of initial conditions and final times in parrallel. In each case rather
than pssing a single initial state and final time we pass a lists of each. The outputs to the call will then be list
of length n containing the outputs of the regular non-parralel mehtod for the ith input state and time.
'''

n = 100
nthreads = 8

X0t0s =[X0t0]*n
tfs   =[tf]*n


Xftfs = TBInteg.integrate_parallel(X0t0s,tfs,nthreads)

Xftf_Jacs = TBInteg.integrate_stm_parallel(X0t0s,tfs,nthreads)

Xftf_Jac_EventLocs = TBInteg.integrate_stm_parallel(X0t0s,tfs,Events,nthreads)

Trajs  = TBInteg.integrate_dense_parallel(X0t0s,tfs,nthreads)

Traj_EventLocs  = TBInteg.integrate_dense_parallel(X0t0s,tfs,Events,nthreads)

for i in range(0,n):
    
    Xftf = Xftfs[i]
    Xftf,Jac = Xftf_Jacs[i]
    Xftf,Jac,EventLocs = Xftf_Jac_EventLocs[i]
    
    Traj = Trajs[i]
    Traj,EventLocs = Traj_EventLocs[i]



######### Local Control Laws  ##############
'''
In the previous examples we only exmined how to integrate ODE's with no control,
or constant controls, but often time we need to compute controls as a function
of the local state or time. We can do this in asset by initializing our integrator with
a control law. As an example,lets resue our twobodyLT ode from the ODEGuide section.
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
        
'''
We will add a control to the integrator specifying that throttle should be 80%
of the maximum and aligned with the spacecraft's intantaneous velocity vector. We do this by first writinh
an ASSET vector function , that is assumed to take only the velocity as aguments and outputs
the desired control vector. We can then pass this to integrator constrauctor along with a list
specifying the indices fullstate varibales we want to forward to out control law. In this case it
is [3,4,5] which are the velocty varaibles as we have defined in our ODE.This control law will now be applied to all of our integrations.
'''        

def ULaw(throttle):
    V = Args(3)
    return V.normalized()*throttle


ode = TwoBodyLTODE(1,.01)

integNoUlaw = ode.integrator("DOPRI87",.1)
integULaw   = ode.integrator("DOPRI87",.1,ULaw(0.8),[3,4,5])



r  = 1.0
v  = 1.1
t0 = 0.0
tf = 20.0


X0t0U0 = np.zeros((10))
X0t0U0[0]=r
X0t0U0[4]=v
X0t0U0[6]=t0        


TrajNoULaw = integNoUlaw.integrate_dense(X0t0U0,tf)

TrajULaw   = integULaw.integrate_dense(X0t0U0,tf)



####################
TT = np.array(TrajNoULaw).T
plt.plot(TT[0],TT[1],label='TrajNoULaw',marker='o')

TT = np.array(TrajULaw).T
plt.plot(TT[0],TT[1],label='TrajULaw',marker='o')

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.axis("Equal")
plt.grid(True)
        
plt.show()












