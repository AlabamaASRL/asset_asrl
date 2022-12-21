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
    VPS0,VPS1,Lvar = Args(27).tolist([(0,13),(13,13),(26,1)])
    return (VPS0-VPS1)*Lvar


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
    VS0,VP1 = Args(8).tolist([(0,4),(4,4)])
    return VS0.dot(VP1)


XtUvars0 = [3,4,5]
SPvars0  = [0]

XtUvars1 = [3,1,2]
OPvars1 = [0]

## Enforce that the dot procuct of the specified variables from each phase region =0
ocp.addLinkEqualCon(ALinkEqualCon(),
                    0,'Last', XtUvars0,[],SPvars0,
                    1,'First',XtUvars1,OPvars1,[])


###############################################################################

SomeFunc = Args(6).head(3).cross(Args(6).tail(3))
## Only need XtUVars from phases 2 and 3 at the specified regions
ocp.addLinkEqualCon(SomeFunc,
                    phase2,'Last', range(0,3),
                    phase3,'First',range(0,3))


SomeOtherFunc = Args(2).sum()-1
# Only needs ODEparams from phases 0 and 1
ocp.addLinkEqualCon(SomeOtherFunc,
                    0,'ODEParams', [0],
                    1,'ODEParams', [0])

##############################################################################


def TriplePhaseLink():
    X0,X1,X2 = Args(9).tolist([(0,3),(3,3),(6,3)])
    
    return vf.sum(X0,X1,X2)


XtUvars = range(3,6)
SPvars = []  # none needed, leave empty but still pass it in
OPvars = []  # none needed, leave empty but still pass it in
LPvars = []  # none needed, leave empty but still pass it in

# List of tuples of the variables and regions needed from each phase
ocp.addLinkEqualCon(TriplePhaseLink(),
                    [(3,'First', XtUvars,OPvars,OPvars),
                     (4,'First', XtUvars,OPvars,OPvars),
                     (5,'First', XtUvars,OPvars,OPvars)],
                    LPvars)
##################################################################


# Enforce that the norm of first 3 link params is 1
LPvec = [0,1,2]
ocp.addLinkParamEqualCon(Args(3).norm()-1.0,LPvec)

# Apply same constraint to multiple groups of 3 link params
LPvecs = [[0,1,2] ,[3,4,5],[6,7,8]]
ocp.addLinkParamEqualCon(Args(3).norm()-1.0,LPvecs)


################################################################

# Enforce that variables XtUvars [3,4,5] in the last state of phase0
# are equal to the same variables in the first state of phase1
ocp.addDirectLinkEqualCon(0,'Last',range(3,6),
                          1,'First',range(3,6))



# Enforce continuity between the last time in phase1 (time is index 7)
# And the first time in phase2 (time is index 6!!)
ocp.addDirectLinkEqualCon(1,'Last',[7],
                          2,'First',[6])


# Enforce that the ODE varaibels in phase 0 ans phase 1 are equal
ocp.addDirectLinkEqualCon(0,'ODEParams',[0],
                          1,'ODEParams',[0])

############################################################################

# Enforce forward time continuity in XtUvars [0,1,2] across all phases
for i in range(0,5):
    ocp.addDirectLinkEqualCon(i,'Last',range(0,3),
                              i+1,'First',range(0,3))

### OR

ocp.addForwardLinkEqualCon(0,5,range(3,6))

#
ocp.addForwardLinkEqualCon(phase0,phase5,range(3,6))
############################################################################
############################################################################

def ALinkInequalCon():
    VS0,VP1 = Args(8).tolist([(0,4),(4,4)])
    return VS0.dot(VP1)


XtUvars0 = [3,4,5]
SPvars0  = [0]

XtUvars1 = [3,1,2]
OPvars1 = [0]

## Enforce that the dot procuct of the specified variables from each phase region <0
ocp.addLinkInequalCon(ALinkInequalCon(),
                    0,'Last', XtUvars0,[],SPvars0,
                    1,'First',XtUvars1,OPvars1,[])



def TriplePhaseInequality():
    X0,X1,X2 = Args(9).tolist([(0,3),(3,3),(6,3)])
    
    return vf.sum(X0,X1,X2)


XtUvars = range(3,6)
SPvars = []  # none needed, leave empty but still pass it in
OPvars = []  # none needed, leave empty but still pass it in
LPvars = []  # none needed, leave empty but still pass it in

# List of tuples of the variables and regions needed from each phase
ocp.addLinkInequalCon(TriplePhaseInequality(),
                    [(3,'First', XtUvars,OPvars,OPvars),
                     (4,'First', XtUvars,OPvars,OPvars),
                     (5,'First', XtUvars,OPvars,OPvars)],
                    LPvars)



######################################################################

# Enforce that the norm of first 3 link params is < 1
LPvec = [0,1,2]
ocp.addLinkParamInequalCon(Args(3).norm()-1.0,LPvec)

# Apply same constraint to multiple groups of 3 link params
LPvecs = [[0,1,2] ,[3,4,5],[6,7,8]]
ocp.addLinkParamInequalCon(Args(3).norm()-1.0,LPvecs)


LinkParams = ocp.returnLinkParams()

##############################################################################
##############################################################################

def ALinkObjective():
    VS0,VP1 = Args(8).tolist([(0,4),(4,4)])
    return VS0.dot(VP1)  # is a scalar function


XtUvars0 = [3,4,5]
SPvars0  = [0]

XtUvars1 = [3,1,2]
OPvars1 = [0]


ocp.addLinkObjective(ALinkObjective(),
                    0,'Last', XtUvars0,[],SPvars0,
                    1,'First',XtUvars1,OPvars1,[])



# Minimize the sum of the norms of these groups of 3 link params

LPvecs = [[0,1,2] ,[3,4,5],[6,7,8]]
ocp.addLinkParamObjective(Args(3).norm(),LPvecs)
