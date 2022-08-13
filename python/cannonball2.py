

# Set up directory structure
import sys 
import os 
sys.path.append('/scmdnav')

# Import asset and necessary tools.
import asset as ast 
vf = ast.VectorFunctions 
Args = vf.Arguments 
oc = ast.OptimalControl
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

# Other function libraries
import numpy as np
import plotly.graph_objects as go 

# Define our ODE.
class CannonballOde(oc.ode_x.ode):
    def __init__(self):
        
        # Constants
        g = 9.81

        args = Args(5)
        r = args.head(2)
        v = args.segment(2,2)
        t = args.tail(1)

        rdot = v 
        vdot = vf.ConstantVector(5, [0, -g]).vf()
        eom = vf.Stack([rdot, vdot])
        oc.ode_x.ode.__init__(self,eom,4)
        return


# ODE and integrator
ode = CannonballOde()
integ = ode.integrator(0.01)

# Initial Guess
vmag = 20
th0 = 80*np.pi/180
X0 = np.array([0,0,vmag*np.cos(th0),vmag*np.sin(th0), 0])
traj_IG = integ.integrate_dense(X0, 4, 400)


# Init constraint on angle
def vel_angle():
    args = Args(3)
    v = args.head(2)
    costh = args[2]
    c1 = vf.dot(v.normalized(), vf.ConstantVector(3, [1,0])) - costh
    return c1

# Create phase
phase = ode.phase(Tmodes.LGL3, traj_IG, 200)
phase.addBoundaryValue(PhaseRegs.Front, [0,1], [0,0]) # Constrain initial position
phase.addBoundaryValue(PhaseRegs.Back, [1], [0]) # Constrain Y position to zero
phase.addLUNormBound(PhaseRegs.Front, [2,3], vmag, vmag) # Initial velocity magnitude
phase.addEqualCon(PhaseRegs.Front,vel_angle(),[2,3],[],[0]) # static param for velocity angle
phase.setStaticParams([np.cos(th0)])
phase.addValueObjective(PhaseRegs.Back, 0, -1.0)
phase.Threads=4
phase.optimizer.PrintLevel = 0
phase.solve_optimize()
traj_opt = phase.returnTraj()

fig = go.Figure(layout={
    'template':'plotly_dark',
    'title_xanchor':'center',
    'title_x':0.5,
    'xaxis_title':'X (m)',
    'yaxis_title':'Y (m)',
    'yaxis_scaleanchor':'x',
    'yaxis_scaleratio':1.0
})


trj = np.array(traj_IG)
fig.add_trace(go.Scatter(
    x = trj[:,0],
    y = trj[:,1],
    mode='lines+markers',
    name='initial guess',
    line={'color':'blue'}
))

trj = np.array(traj_opt)
fig.add_trace(go.Scatter(
    x = trj[:,0],
    y = trj[:,1],
    mode='lines+markers',
    name='optimized',
    line={'color':'red'}
))



fig.write_html('cannonball.html',config={'editable':True})






