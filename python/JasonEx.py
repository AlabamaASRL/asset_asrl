import numpy as np 
import asset as ast 
import plotly.graph_objects as go

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def TestODE():
    args = vf.Arguments(4)
    x = args[0]
    v = args[1]
    u = args[3] # Time is idx 2
    return vf.Stack([v, u])

def TestObj():
    args = vf.Arguments(1)
    u = args[0]
    return u*u

nseg = 200 # 100 segs will diverge
ode_1d = oc.ode_x_x.ode(TestODE(),2,1)
phase = ode_1d.phase(Tmodes.LGL3)
phase.setControlMode(oc.ControlModes.BlockConstant)

ig = np.vstack((
    np.linspace(0.0, 1.0, nseg), 
    np.ones((nseg,)), 
    np.linspace(0.0, 1.0, nseg), 
    np.ones((nseg,))
)).T
phase.setTraj(ig, nseg)
phase.addBoundaryValue(PhaseRegs.Front,[0,1,2],[0,0,0])
phase.addBoundaryValue(PhaseRegs.Back, [0,1,2],[1,0,1])
phase.addIntegralObjective(TestObj(),[3])
phase.optimizer.OptLSMode = ast.LineSearchModes.L1
#phase.solve_optimize()

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

traj = np.array(phase.returnTraj())
fig = go.Figure()
fig.add_trace(go.Scatter( 
    x = traj[:,2],
    y = traj[:,0],
    mode='lines+markers',
    name='position'
 ))
fig.add_trace(go.Scatter( 
    x = traj[:,2],
    y = traj[:,1],
    mode='lines+markers',
    name='velocity'
 ))
fig.show()
