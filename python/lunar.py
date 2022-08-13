import sys
sys.path.append('..')

import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from numpy import cos,sin,tan,arccos,arcsin,arctan2
from scipy.spatial.transform import Rotation as R

import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "chrome"

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes

class LunarDynamics(oc.ode_x_x.ode):
    def __init__(self, mu, g0, Tmax, isp):
        
        # Input
        Xvars = 7
        Uvars = 3
        Ivars = Xvars + 1 + Uvars 
        args = Args(Ivars)
        r = args.head_3()
        v = args.segment_3(3)
        m = args[6]
        u = args.tail_3()
        eta = u.norm()

        a_grav = -mu * r.normalized_power3()
        a_thr = Tmax * u/m
        
        # Output
        rdot = v 
        vdot = a_grav + a_thr
        mdot = Tmax * eta / ( g0 * isp )
        xdot = vf.Stack([rdot,vdot,mdot])

        super().__init__(xdot,Xvars,Uvars)


def normalize_system(mu, r, mstar):
    units = {}
    units["astar"] = mu/r**2
    units["lstar"] = r
    units["vstar"] = np.sqrt(mu / r)
    units["tstar"] = np.sqrt(r**3/mu)
    units["mustar"] = mu
    units["mstar"] = mstar
    return units


# Normalize the system.
R_M = 1737.1
mu_M = 4.9048695e3
m0 = 30000
isp = 250
Tmax = 10000
units = normalize_system(mu_M, R_M, m0)
KM = 1.0 / units["lstar"]
KM_S = 1.0 / units["vstar"]
M = 0.001 * KM
M_S = 0.001 * KM_S
S = 1.0 / units["tstar"]
N = 1.0 / (units["mstar"]*M/(S**2))
g0 = 9.801
Day = 86400.0 * S
mu_nd = 1
isp_nd = isp * S
Tmax_nd = Tmax * N
m0_nd = m0 / units["mstar"]
print(Tmax_nd)
print(Tmax_nd)

# Construct initial orbit.
# Ask James for integration boundaries. Perhaps LGL interp table interpolation
# post integration is sufficient.

ra = (R_M + 100)*KM
rp = (R_M + 10)*KM
a = 0.5*(ra + rp)
e = (ra - rp)/(ra + rp)
p = a*(1 - e**2)
h = np.sqrt(mu_M*p)
print(mu_nd)
# Create ode.
ode = LunarDynamics(mu_nd, g0, Tmax_nd, isp_nd)

# Define initial state
a_0 = a
e_0 = e
i_0 = 90 * np.pi / 180
Om_0 = 90 * np.pi / 180
om_0 = -90 * np.pi / 180
nu_0 = -42 * np.pi / 180
i = i_0
Om = Om_0
om = om_0
nu = nu_0
E_0 = 2 * np.arctan(np.sqrt(1-e_0)/np.sqrt(1+e_0)*tan(nu_0/2))
rc = a_0*(1-e_0*np.cos(E_0))
o = rc*np.array([np.cos(nu_0), np.sin(nu_0), 0])
od = np.sqrt(mu_nd*a)/rc*np.array([-np.sin(E_0),np.sqrt(1-e_0**2)*np.cos(E_0),0])
R1 = R.from_dcm([[cos(-om),sin(-om),0],[-sin(-om),cos(-om),0],[0,0,1]])
R2 = R.from_dcm([[1,0,0],[0,cos(-i),sin(-i)],[0,-sin(-i),cos(-i)]])
R3 = R.from_dcm([[cos(-Om),sin(-Om),0],[-sin(-Om),cos(-Om),0],[0,0,1]])
R_orb = R3 * R2 * R1
r0 = R_orb.apply(o)
v0 = R_orb.apply(od)
X0 = r0.tolist() + v0.tolist() + [m0_nd] + [0]+ [.01,.01,.01]

# Propagate initial guess
def NoControl():
    args = Args(6)
    rhat = args.head(3)
    vhat = -args.tail(3)
    return (.8*vhat + .37*rhat).normalized()
integ_coast = ode.integrator(10*S, NoControl(), range(0,6))

coast_traj = integ_coast.integrate_dense(X0, 900*S, 200)
init_guess = coast_traj
print(len(init_guess[0]))
# Create optimization
phase = ode.phase(Tmodes.LGL3,init_guess,len(init_guess))

phase.addBoundaryValue(PhaseRegs.Front,range(0,7),X0[0:7])
phase.addBoundaryValue(PhaseRegs.Back,range(0,3),[0,0,0])
phase.addEqualCon(PhaseRegs.Back,Args(3).norm()-R_M*KM,[0,1,2])
phase.Threads=8
phase.optimizer.QPThreads=6
phase.optimizer.PrintLevel=0
phase.addLUVarBound(PhaseRegs.Path,6,0.1,1)
phase.addLUNormBound(PhaseRegs.Path,[8,9,10],.001,1.0,1.0)
# phase.addStateObjective(PhaseRegs.Back, 6)
phase.addDeltaTimeObjective(1)
phase.solve()
pt = phase.returnTraj()

#PT = np.array()





def plot_swirl(fig,rad,nsc,nlat,nlon):
    for rot in np.linspace(0,2*np.pi,nlon)[:-1]:
        R1 = np.array([[cos(rot),sin(rot),0],[-sin(rot),cos(rot),0],[0,0,1]])
        pts = []
        for rad in np.linspace(0,4*np.pi,nsc):
            R2 = np.array([[1,0,0],[0,cos(rad),sin(rad)],[0,-sin(rad),cos(rad)]])
            pts.append(np.matmul( R2.T, np.matmul(R1.T,[0,rad,0]) ))
        fig.add_trace(go.Scatter3d(
            x = [X[0] for X in pts],
            y = [X[1] for X in pts],
            z = [X[2] for X in pts],
            mode='lines',
            line_color='white'
        ))

def plot_sphere(fig,r,nsc,nlat,nlon):
    for rot in np.linspace(0,2*np.pi,nlat)[:-1]:
        R1 = np.array([[cos(rot),sin(rot),0],[-sin(rot),cos(rot),0],[0,0,1]])
        pts = []
        for rad in np.linspace(0,2*np.pi,nsc):
            R2 = np.array([[1,0,0],[0,cos(rad),sin(rad)],[0,-sin(rad),cos(rad)]])
            R = np.matmul(R1,R2)
            x_r2 = np.matmul(R, [0,r,0])
            pts.append(x_r2)
        fig.add_trace(go.Scatter3d(
            x = [X[0] for X in pts],
            y = [X[1] for X in pts],
            z = [X[2] for X in pts],
            mode='lines',
            line_color='gray'
        ))

    for rot in np.linspace(-np.pi,np.pi,nlon)[:-1]:
        R1 = np.array([[1,0,0],[0,cos(rot),sin(rot)],[0,-sin(rot),cos(rot)]])
        pts = []
        for rad in np.linspace(0,2*np.pi,nsc):
            R2 = np.array([[cos(rad),sin(rad),0],[-sin(rad),cos(rad),0],[0,0,1]])
            R = np.matmul(R2,R1)
            x_r2 = np.matmul(R, [0,r,0])
            pts.append(x_r2)
        fig.add_trace(go.Scatter3d(
            x = [X[0] for X in pts],
            y = [X[1] for X in pts],
            z = [X[2] for X in pts],
            mode='lines',
            line_color='gray'
        ))

fig = go.Figure(layout={'template':'plotly_dark','scene_aspectmode':'data'})
fig.add_trace(go.Scatter3d(
    x = [X[0]/KM for X in coast_traj],
    y = [X[1]/KM for X in coast_traj],
    z = [X[2]/KM for X in coast_traj],
    mode='lines',
    line_color='white'
))
fig.add_trace(go.Scatter3d(
    x = [X[0]/KM for X in pt],
    y = [X[1]/KM for X in pt],
    z = [X[2]/KM for X in pt],
    mode='lines',
    line_color='red'
))
plot_sphere(fig,R_M,200,60,60)
fig.write_html("output.html")


