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




ode = TwoBodyLTODE(1,.01)

r  = 1.0
v  = 1.1
t0 = 0.0
tf = 20.0


X0t0U0 = np.zeros((10))
X0t0U0[0]=r
X0t0U0[4]=v
X0t0U0[6]=t0        


def ULaw(throttle):
    V = Args(3)
    return V.normalized()*throttle


integULaw   = ode.integrator("DP54",.1,ULaw(0.8),[3,4,5])

TrajI   = integULaw.integrate_dense(X0t0U0,tf,6000)

# Construct from an ode,its,dimensions,and a trajectory of the correct size
# Most accurate interpolation
Tab1 = oc.LGLInterpTable(ode.vf(),6,3,TrajI)


## Construct from arbitrary time series data,
## Elements constist of data followed by time
## No ode needed, but less accurate interpolation
JustUts = [ list(T[7:10]) +[T[6]] for T in TrajI  ]
Tab2 = oc.LGLInterpTable(JustUts)


# Interpolation returns all data stored in the table, including time

print(Tab1(0.0)) # prints [1.,  0.,  0.,  0.,  1.1, 0.,  0.,  0.,  0.8, 0. ]

print(Tab2(0.0)) # prints [0.,  0.8, 0.,  0. ]

#############################################################

# Tables consisting of full trajectories of the right size will be interpreted 
# to use the controls as a time dependent control law
integTab1 = ode.integrator(.1,Tab1)

## If the data is not the same size as an ODE input you should manually
## Specify which elements of the outputs of the table should be controls
## Since Tab1 is the right size, this does the same thing as above
integTab1 = ode.integrator(.1,Tab1,range(7,10))

# However, Tab2 is just controls so we need to specify which elements of 
# the output of the table are the controls
integTab2 = ode.integrator(.1,Tab2,range(0,3))

Traj1   = integTab1.integrate_dense(X0t0U0,tf)
Traj2   = integTab2.integrate_dense(X0t0U0,tf)



##################################

def RendFunc(Tab):
    X,t = Args(7).tolist([(0,6),(6,1)])
    
    # Convert table into a vector function
    # that takes a time and returns the specified elements in the table
    # in this case just, position and velocity
    X_tfunc = oc.InterpFunction(Tab,range(0,6))
    
    return X-X_tfunc(t)
    
RFunc = RendFunc(Tab1)

print(RFunc(TrajI[10][0:7]))  # prints [0,0,0,0,0,0]

#########################################################











####################
#TT = np.array(TrajNoULaw).T
#plt.plot(TT[0],TT[1],label='TrajNoULaw',marker='o')

TT = np.array(TrajI).T
plt.plot(TT[0],TT[1],label='TrajULaw',marker='o')

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.axis("Equal")
plt.grid(True)
        
plt.show()
