import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes


class SSTO(oc.ode_x_u.ode):
    def __init__(self,F_T,Isp,g,rho_ref,h_scale,CDA):
        ############################################################
        args  = oc.ODEArguments(5,1)
        X = args.XVec()
        U = args.UVec()
        
        y     =X[1]
        vx   = X[2]
        vy   = X[3]
        m    = X[4]
        theta =U[0]
        
        rho = rho_ref * np.exp(-y / h_scale)
        xdot = vx
        ydot = vy
        vxdot = (F_T * vf.cos(theta) - (0.5 * CDA) * rho * vx**2) / m
        vydot = (F_T * vf.sin(theta) - (0.5 * CDA) * rho * vy**2) / m - g
        mdot = -F_T / (g * Isp)
        ode = vf.stack(xdot,ydot,vxdot,vydot,mdot)
        ##############################################################
        super().__init__(ode,5,1)
        
        
Lstar = 185000.0
Tstar = 60.0
Mstar = 117000

Vstar = Lstar/Tstar
Fstar = Mstar*Lstar/(Tstar**2)
Astar = Lstar/(Tstar**2)
Rhostar = Mstar/(Lstar**3)


g       = 9.80665   /Astar
rho_ref = 1.225     /Rhostar
h_scale = 8.44e3    /Lstar
CDA     = 0.5*7.069 /(Lstar**2)
F_T     = 2100000.0 /Fstar
Isp     = 265.2     /Tstar
m0      = 117000    /Mstar
yf      = 185000.0  /Lstar
vxf     = 7796.6961 /Vstar

tf     = 120.0/Tstar
tpitch = 90.0/Tstar


ode = SSTO(F_T,Isp,g,rho_ref,h_scale,CDA)


def UCon():
    t = Args(1)[0]
    f = vf.ifelse(t<tpitch, 0*t + np.pi/2, 0*t  )
    return f

IState =np.zeros((7))
IState[4] =m0
IState[6] =np.pi/2.0
integ  = ode.integrator(tf/1000.0,UCon(),[5])
TrajIG = integ.integrate_dense(IState,tf,300)

phase = ode.phase(Tmodes.LGL3,TrajIG,128)
#phase.setControlMode(Cmodes.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front,range(0,6),IState[0:6])
phase.addBoundaryValue(PhaseRegs.Back ,[1,2,3],[yf,vxf,0])
phase.addLUVarBound(PhaseRegs.Path,6,-np.pi/1.99,np.pi/1.99,1.0)
phase.addDeltaTimeObjective(1)
phase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
phase.optimizer.MaxLSIters = 2
phase.optimizer.PrintLevel =1
phase.optimize()


TT = np.array(phase.returnTraj()).T

print(TT[5][-1]*Tstar)

plt.plot(TT[0],TT[1],marker='.')
plt.axis("Equal")

plt.show()

plt.plot(TT[5],TT[2])
plt.plot(TT[5],TT[3])
plt.show()

plt.plot(TT[5],TT[6])
plt.show()





