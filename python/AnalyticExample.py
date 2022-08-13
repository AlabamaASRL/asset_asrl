import numpy as np
import asset as ast
import matplotlib.pyplot as plt


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags



class Example1ODE(oc.ode_x_u.ode):
    def __init__(self):
        args = oc.ODEArguments(1,1)
        x = args.XVar(0)
        u = args.UVar(0)
        xdot = .5*x + u
        super().__init__(xdot,1,1)

    class obj(vf.ScalarFunction):
        def __init__(self):
            args = Args(2)
            x=args[0]
            u=args[1]
            obj = u*u + x*u + 1.25*x**2
            super().__init__(obj)
        

##############################################
ode = Example1ODE()

x0 = 1.0
t0 = 0.0
tf = 1.0
u0 = .01

nsegs = 64
method = Tmodes.LGL3

TrajIG = [[x0,t,u0] for t in np.linspace(t0,tf,100)]

phase = ode.phase(method)

phase.integrator = ode.integrator(.0005)
phase.integrator.Adaptive = True
phase.integrator.FastAdaptiveSTM=True
phase.integrator.setAbsTol(1.0e-10)
phase.enable_vectorization(True)
phase.setTraj(TrajIG,nsegs)
#phase.setControlMode(oc.ControlModes.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front,[0,1],[x0,t0])
phase.addBoundaryValue(PhaseRegs.Back, [1],  [tf])
phase.addIntegralObjective(Example1ODE.obj(),[0,2])
phase.Threads=4
phase.optimizer.KKTtol =1.0e-9
#phase.solve()
phase.optimize()
Traj = phase.returnTraj()
CTraj= phase.returnCostateTraj()

###########################################
T = np.array(Traj).T
CT = np.array(CTraj).T

X = T[0]
t = T[1]
U = T[2]
### Analytic costates
lstar = 2*np.cosh(1-t)*np.tanh(1-t)/np.cosh(1)
lm = CT[0]
##########################################
plt.plot(t,lm,label   ='Collocation')
plt.plot(t,lstar,label='Analytic')

plt.show()
#################################
    
    