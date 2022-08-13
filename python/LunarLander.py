import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from numpy import cos,sin,tan,arccos,arcsin,arctan2
from scipy.spatial.transform import Rotation as R
import MKgSecConstants as c

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes


class PlanarLander(oc.ode_x_u.ode):
    def __init__(self,mu,Rbod,Ascale,Mscale, ACSscale, Iscale):
        args = oc.ODEArguments(7,2)
        alt = args[0]
        drange = args[1]
        vr = args[2]
        vt = args[3]
        m  = args[4]
        angvel = args[5]
        theta = args[6]
        
        thrust = args.UVec()[0]
        torque = ACSscale * args.UVec()[1]
        
        Tr = (Ascale*thrust/m)*vf.sin(theta)
        Tt = (Ascale*thrust/m)*vf.cos(theta)
        
        I = m*Iscale
        r = alt + Rbod
        
        alt_dt    = vr
        drange_dt = vt
        vr_dt     = (vt**2)/r - mu/(r**2) + Tr
        vt_dt     =  - vr*vt/r + Tt
        mdot      =  - thrust*Mscale
        angacc    = torque*(1.0/I) -vr*vt/r**2
        
        xdot = vf.stack([alt_dt,drange_dt,vr_dt,vt_dt,mdot, angacc, angvel])
        super().__init__(xdot,7,2)




# using appollo specs
MTotal   = 16000 #Kg
Tmax     = 45000 #N 

Length = 7.04
ACSmax = 150*750 #N-m
Isquare = (1/12.0)*MTotal*(Length**2 + Length**2) #kg/m^2


ISP      = 311   #
mdot     = Tmax/(9.81*ISP)
print(mdot)
h0       = 15000   #m
Rmoon    = 1736000 #m
Vt0      = np.sqrt(c.MuMoon/(Rmoon + h0))


lstar     = Rmoon
vstar     = np.sqrt(c.MuMoon/lstar)
tstar     = np.sqrt(lstar**3/c.MuMoon)
astar     = c.MuMoon/lstar**2
mstar     = MTotal
mdstar    = mstar/tstar

    


print("lstar: ", lstar," m")
print("tstar: ", tstar," s")

print("vstar: ", vstar," m/s")
print("astar: ", astar," m/s^2")
print("mdstar: ", mdstar," Kg/s")

mustar = (lstar**3)/(tstar**2)

mu_nd     = 1
Rbod_nd   = Rmoon/lstar
Ascale_nd = (Tmax/MTotal)/astar
Mscale_nd = mdot/mdstar
ACSscale_nd = ((ACSmax/MTotal)/astar)/Length
Iscale_nd = (Isquare/MTotal)/(Length**2)

print(ACSscale_nd, Iscale_nd)

h0_nd     = h0/lstar
v0_nd     = Vt0/vstar


print("mu_nd: ", mu_nd)
print("g_nd: ", (c.MuMoon/Rmoon**2)/astar)
print("a_nd: ", Ascale_nd)
print("a_nd: ", h0_nd)

print("Mscale_nd: ", Mscale_nd)


ode = PlanarLander(mu_nd,Rbod_nd,Ascale_nd,Mscale_nd, ACSscale_nd, Iscale_nd)

IG =np.array([h0_nd,0,0,v0_nd,1.0, 0, np.pi, 0,.8,-5.])

integ = ode.integrator(.01)

initTraj = integ.integrate_dense(IG,.3,250)

TT = np.array(initTraj).T
plt.plot(TT[7]*tstar,TT[0]*lstar/1000)
plt.ylabel("Altitude (Km)")
plt.xlabel("time (sec)")
plt.grid(True)
plt.show()

plt.plot(TT[7]*tstar,TT[2])
plt.plot(TT[7]*tstar,TT[3])
plt.show()

plt.plot(TT[7]*tstar,TT[8])
plt.show()

plt.plot(TT[7]*tstar,TT[6])
plt.show()

phase = ode.phase(Tmodes.LGL3,initTraj,300)
phase.addBoundaryValue(PhaseRegs.Front,range(0,6),IG[0:6])
phase.addBoundaryValue(PhaseRegs.Front,[7],[0.])

phase.addBoundaryValue(PhaseRegs.Back,[0,2, 3],[0,0,0])

phase.addLUVarBound(PhaseRegs.Path,0, h0_nd*(-.01),h0_nd*1.5,1.0)
phase.addLUVarBound(PhaseRegs.Path,8, 0.0001,1.0,1.0)
phase.addLUVarBound(PhaseRegs.Path,9,1.0,-1.,1.0)

input("S")
phase.addDeltaTimeObjective(1.0)
#phase.addValueObjective(PhaseRegs.Back, 4, -1.0)
phase.optimizer.deltaH =1.0e-6

phase.optimizer.MaxAccIters = 150
phase.optimizer.OptLSMode = ast.Solvers.LineSearchModes.L1
phase.optimize()
phase.addBoundaryValue(PhaseRegs.Back,[6],[np.pi/2.0])
phase.optimize()

Traj = phase.returnTraj()

TT = np.array(Traj).T


plt.plot(TT[7]*tstar,TT[0]*lstar/1000)
plt.ylabel("Altitude (Km)")
plt.xlabel("time (sec)")
plt.grid(True)
plt.show()

plt.plot(TT[1]*lstar/1000,TT[0]*lstar/1000)
plt.ylabel("Altitude (Km)")
plt.xlabel("Downrange (Km)")
plt.show()


plt.plot(TT[7]*tstar,TT[2])
plt.plot(TT[7]*tstar,TT[3])
plt.show()

plt.plot(TT[7]*tstar,TT[8])
plt.show()

plt.plot(TT[7]*tstar,TT[4])
plt.show()


plt.plot(TT[7]*tstar,TT[6])
plt.show()





