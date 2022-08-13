import numpy as np
import asset as ast

vf  = ast.VectorFunctions
oc  = ast.OptimalControl
sol = ast.Solvers

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes

PhaseRegs = oc.PhaseRegionFlags

    
class LTModel(oc.ode_x_u.ode):
    def __init__(self,mu,ltacc):
        
        Xvars = 6
        Uvars = 3
        ############################################################
        args = oc.ODEArguments(Xvars,Uvars)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        g = r.normalized_power3()*(-mu)
        thrust = u*ltacc
        acc = g + thrust
        ode =  vf.Stack([v,acc])
        #############################################################
        super().__init__(ode,Xvars,Uvars)

    class massobj(vf.ScalarFunction):
        def __init__(self,scale):
            u = Args(3)
            super().__init__(u.norm()*scale)
    class powerobj(vf.ScalarFunction):
        def __init__(self,scale):
            u = Args(3)
            super().__init__(u.norm().squared()*scale)



mu = 1
acc = .02
ode    = LTModel(mu,acc)

r0 = 1.0
v0 = np.sqrt(mu/r0)

rf = 2.0
vF = np.sqrt(mu/rf)

X0 = np.zeros((7))
X0[0]=r0
X0[4]=v0

Xf = np.zeros((6))
Xf[0]=rf
Xf[4]=vF

XIG = np.zeros((10))
XIG[0:7]=X0
XIG[7] =.99

#ode.vf().rpt(XIG,1000000)

#ode.vf().SuperTest(XIG,10000000)


integ  = ode.integrator(.01,Args(3).normalized()*.8,[3,4,5])
TrajIG = integ.integrate_dense(XIG,6.4*np.pi,100)

for i in range(0,1):
    phase = ode.phase(Tmodes.LGL3,TrajIG,512)
    phase.optimizer.deltaH=1.0e-6
    #phase.setControlMode(Cmodes.BlockConstant)
    phase.addBoundaryValue(PhaseRegs.Front,range(0,7),X0)
    phase.addLUNormBound(PhaseRegs.Path,[7,8,9],.001,1.0,1.0)
    phase.addBoundaryValue(PhaseRegs.Back,range(0,6),Xf[0:6])
    phase.optimizer.QPThreads=8
    phase.Threads=8
    #phase.optimizer.QPOrderingMode = ast.QPOrderingModes.MINDEG
    phase.optimizer.PrintLevel=1
    ########################################
    phase.addDeltaTimeObjective(1.0)
    
    phase.optimize()
    
    phase.removeStateObjective(-1)
    ########################################
    phase.addIntegralObjective(LTModel.powerobj(0.5),[7,8,9])
    phase.optimize()
    
    phase.removeIntegralObjective(-1)
    #######################################
    
    phase.addIntegralObjective(LTModel.massobj(1.0),[7,8,9])
    phase.optimize()
    print(i)

print("Finished")






