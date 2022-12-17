import numpy as np
import asset_asrl as ast


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Args      = vf.Arguments


class DummyODE(oc.ODEBase):
        def __init__(self,xv,uv,pv):
            args = oc.ODEArguments(xv,uv,pv)
            super().__init__(args.XVec(),xv,uv,pv)
        
        
# 6 states indexed [0,1,2,3,4,5], time index 6, no controls or ODE params
odeX   = DummyODE(6, 0, 0)

# 6 states indexed [0,1,2,3,4,5], time index 6, 3 controls [7,8,9], no ODE params
odeXU  = DummyODE(6, 3, 0)

# 7 states indexed [0,1,2,3,4,5,6], time index 7, 3 controls indexed [8,9,10], one ODE param
odeXUP = DummyODE(7, 3, 1)


phase0 = odeXUP.phase("LGL3")
phase1 = odeXUP.phase("LGL3")

phase0.setStaticParams([0.0])
phase1.setStaticParams([0.0])

phase2 = odeXU.phase("LGL3")
phase3 = odeXU.phase("LGL3")

phase4 = odeX.phase("LGL3")
phase5 = odeX.phase("LGL3")







ocp  = oc.OptimalControlProblem()

ocp.addPhase(phase0)
ocp.addPhase(phase1)
ocp.addPhase(phase2)
ocp.addPhase(phase3)
ocp.addPhase(phase4)
ocp.addPhase(phase5)

#ocp.addPhase(phase5)  #phases must be unique, adding same phase twice will throw error

###############################################################################
'''
The individual phases in an ocp, MUST Be unique objects. The software will detect
if you attempt to add the same phase to an ocp twice and throw an error. The commented
out line below will throw an error because the specific phase5 object has already been added to
the ocp.
'''

#ocp.addPhase(phase5)  #phases must be unique, adding same phase twice will throw error

###############################################################################
'''
You can access the phases in an ocp using the ocp.Phase(i) method where i
is the index of the phase in the order they were added. If the phase is created
elswhere in the script you can maninuplaute it throught that orbject or via
the .Phase(i) method as shown below. Note, phases are large stateful objects and we
do not make copies of them by default, thus ocp.Phase(0) and phase0 are the EXACT
same object. Be careful not to apply duplicate constraints to the same phase accidentally
as WE DO NOT CHECK FOR THIS
'''
ocp.Phase(0).addBoundaryValue("Front",range(0,6),np.zeros((6)))
'''
Equivalent to above,make sure you dont accidentally do both.
'''
# phase0.addBoundaryValue("Front",range(0,6),np.zeros((6)))


for phase in ocp.Phases:
    
    phase.addDeltaTimeObjective(1.0)

###############################################################################



ocp.setLinkParams(np.ones((15)))

###############################################################################

def ALinkEqualCon():
    V0,V1,Lvar = Args(27).tolist([(0,13),(13,13),(26,1)])
    return (V0-V1)*Lvar


XtUvars0 = range(0,11)
OPvars0 = [0]
SPvars0 = [0]

XtUvars1 = range(0,11)
OPvars1 = [0]
SPvars1 = [0]

LPvars  = [0]

## Use index in the of the phase in the ocp to specify each phase
ocp.addLinkEqualCon(ALinkEqualCon(),
                    0,'Last', XtUvars0,OPvars0,SPvars0,
                    1,'First',XtUvars1,OPvars1,SPvars1,
                    LPvars)

## Same as above, but use the phase objects themselves to specify each phase
ocp.addLinkEqualCon(ALinkEqualCon(),
                    phase0,'Last', XtUvars0,OPvars0,SPvars0,
                    phase1,'First',XtUvars1,OPvars1,SPvars1,
                    LPvars)
    
## Same as above
ocp.addLinkEqualCon(ALinkEqualCon(),
                    ocp.Phase(0),'Last', XtUvars0,OPvars0,SPvars0,
                    ocp.Phase(1),'First',XtUvars1,OPvars1,SPvars1,
                    LPvars)


###############################################################################

def ALinkEqualCon():
    V0,V1 = Args(26).tolist([(0,13),(13,13)])
    return (V0-V1)


XtUvars0 = range(0,11)
OPvars0 = [0]
SPvars0 = [0]

XtUvars1 = range(0,11)
OPvars1 = [0]
SPvars1 = [0]


## Use index in the of the phase in the ocp to specify each phase
ocp.addLinkEqualCon(ALinkEqualCon(),
                    0,'Last', XtUvars0,OPvars0,SPvars0,
                    1,'First',XtUvars1,OPvars1,SPvars1)


