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
        
        
        
odeX   = DummyODE(6, 0, 0)
odeXU  = DummyODE(6, 3, 0)
odeXUP = DummyODE(7, 3, 1)

phase0 = odeX.phase("LGL3")
phase1 = odeX.phase("LGL3")

phase2 = odeXU.phase("LGL3")
phase3 = odeXU.phase("LGL3")

phase4 = odeXUP.phase("LGL3")
phase5 = odeXUP.phase("LGL3")

phase4.setStaticParams([0])
phase5.setStaticParams([0])


ocp  = oc.OptimalControlProblem()

ocp.addPhase(phase0)
ocp.addPhase(phase1)
ocp.addPhase(phase2)
ocp.addPhase(phase3)
ocp.addPhase(phase4)
ocp.addPhase(phase5)

ocp.setLinkParams(np.ones((15)))



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


###############################################################################


'''
By far the most common type of phase linking in optimal control problems that we regularly
encounter is enforcing continuity of certain state variables across a sequential range of phases in an optimal control
problem. This means that starting at some phase we to enforce that the state variables at the back of
one phase are equal to the state variables at the front of the next phase, and then repeat this pattern across
a subset of sequential phases. This can be accomplished in one line using ocp.addForwardLinkEqualCon
as shown below, which enforces Back to Front continiuty in state variables 0,1,2, and across all phases.  
The first two arguments specify the beginning and end of the range of phases over which we want to apply
this equality constraint. The third argument is a list of the indices of the state variables from each phase we are
enforcing conituity over,and are assumed to be same for all phases. You can use either integer indices of the phases or the phases
themselves to specify the beginning and end of the range
'''

ocp.addForwardLinkEqualCon(0,5,[0,1,2])
#ocp.addForwardLinkEqualCon(phase0,phase5,[0,1,2]) #equivalent to above


###############################################################################

'''
However, often times, multiphase optimal control problems are composed of multiple phase/odes with the same state variables, 
but with differing indices inside each phase/ode, thus precludeing the use of only a single call to .addForwardLinkEqualCon. 
To examine how we reslove this issue, lets look at enforcing time continuity across all phases in our optimal control
problem. The first four phases (0->3) all have the time coordinate as the 6th state variable, thus on this range we can do the following.
'''
ocp.addForwardLinkEqualCon(phase0,phase3,[6])
'''
Likewise, phases 4 and 5 have time as the 7th state variable, so we can do this.
'''
ocp.addForwardLinkEqualCon(phase4,phase5,[7])
'''
However,we still have to link times between phase3 (Var 6) and phase4 (Var 7). To do this we will use the more versatile 
ocp.addDirectLinkEqualCon as shown below. In english, this contraint is saying to make state variable 6 in the Back state of phase3
equal to state variable 7 in the Front state of phase4.
'''
ocp.addDirectLinkEqualCon(phase3,"Back",[6],phase4,"Front",[7])

'''
For context, we could if we wanted to, have impelmented  the behavior of
ocp.addForwardLinkEqualCon(0,5,[0,1,2]) using addDirectLinkEqualCon in a loop as shown below

for i in range(0,4):
    ocp.addDirectLinkEqualCon(i,PhaseRegs.Back,[0,1,2],i+1,PhaseRegs.Front,[0,1,2])
'''

##############################################################################
'''
Another common and use of addDirectLinkEqualCon is to enforce equality of Static or ODEParameters across phases. This will likely be neccessary
in any multiphase problems where each phase has their own internal copy of the params for use in the dynamics and constraints but we need to ensure
that their values are equal across all phases. As an example, we can enforce that the single ODEParam in phase4 and phase5 are equal using the line below.
If wer instead wanted to do the static params we would change the corresponding phase region flag
'''
ocp.addDirectLinkEqualCon(phase4,"ODEParams",[0],phase5,"ODEParams",[0])

'''
All examples thus far have assumned that we wanted to just a apply simple equality constraints saying that states in different phases must be
equal to one another, however often we need to express more compilicated equality constraints. This too can be done using addDirectLinkEqualCon by supplying
the specific constraint we want to apply. Lets say, that for some reason we wanted to enforce that the vector of state varibales [3,4,5] in the back state of
phase2 must be orthognal to the same state variables in the first state of phase3.
To illustrate an important a point in just a moment we assume that out problem rquired us 
to also shift the vector of arguments from phase2 by an arbitrary constant.
This can be accomplished  using a custom vectorfunction and addDirectLinkEqualCon as shown below.
'''

def OrthoCon():
    VB2_VF3 = Args(6)
    VB2 = VB2_VF3.head(3) - [0,0,1]
    VF3 = VB2_VF3.tail(3)
    return VB2.dot(VF3)
    
    
ocp.addDirectLinkEqualCon(OrthoCon(),
                          phase2,"Back", [3,4,5],
                          phase3,"Front",[3,4,5])
'''
In english, what this is doing is collecting state variables 3,4,5 from the corresponding regions of both phases IN THE ORDER SPECIIED into a single
list and forwarding them as arguments to the OrthoCon vector function. OthoCon is written under the assumption 
that the vector of vars from phase2 are the first 3 arguments and those from phase3 are the last 3. 

Note that optimal control problems have no knowledge of the internal structure of custom vector functions and vice versa, 
so it is entirely up to the user to ensure that their custom vector functions get forwared arguments in the desired order. 
The only check that we make is that we are forwardsing the correct number of arguments to OrthoCon (6)
For example, say that we made a mistake above and instead wrote this, 

ocp.addDirectLinkEqualCon(OrthoCon(),phase3,"Front",[3,4,5],phase2,"Back",[3,4,5])

ie transposing the order of phase2 and phase3

We are still forwarding the correct states to our function, but it is no longer doing what we think it should be. It is now actually going to forward the
vector from the back of phase3 as the first 3 arguments and those will have the shift applied, the opposite of what we initially intended.

'''
###############################################################################

'''
The above examples likely encompass a large fraction of the link constraints we encouner, 
but we still have not covered all of the tools needed to exprees arbitary link constraints and objectives. For that we will need to
examine the general addLinkEqualCon method. Note the most of what we say here also applies to the addLinkInequalCon and addLinkObjective methods.
'''


'''
Lets return to our OrthoCon example from above, and say that instead of applying that constraint jsut between phases 2 and 3, we now want to
apply it between all sequential phases (phase0 & phase1 , phase1 & phase2, etc.). Obviously, we could just run that code in a loop and revert to integer base indexing
of the phases but that might not always be convenient. We can instead accomplish this with the simplified addLinkEqualCon as shown below. In english, this constraint is saying:
For every pair of phases in the phases_to_link vector, collect varibles from 3,4,5 from the Back of the first listed phase and as well as variables 3,4,5 from the second listed phase
and forward them to  OrthoCon(). Back to front linking over the pair of phases is specified by the LinkFlag or string "BackToFront". 
Other options include "FrontToBack","FrontToFront"etc. 
'''
phases_to_link = [ [i,i+1] for i in range(0,4)]
ocp.addLinkEqualCon(OrthoCon(),"BackToFront",phases_to_link,[3,4,5])

'''
The behavior of any the previous linking examples can also be (and in fact are internally) 
implemented  using the completely general form of addLinkEqualCon As illustrated below. 
Here we manually specify which phases regions,phase_regs, and 
state variables,xtuvars, we will be collecting for the corresping 
phase in each element of the phases_to_link vector. Our constraint does not require any ODE or static
parameters from the phases so these vectors can be left empty. 
Likewise, it does not equire any link parameters defined at 
the optimal control problem so lpvars can be left blank as well.

'''

phase_regs           = ["Back","Front"]
xtuvars              = [[3,4,5],[3,4,5]]
opvars =[]
spvars =[]
lpvars =[]

ocp.addLinkEqualCon(OrthoCon(),phase_regs,phases_to_link,xtuvars,opvars,spvars,lpvars)

'''
To Examine how, opvars, and spvars, work lets now use the general form of addLinkEqualCon to
write a constraint continity in the 6th state variable of phases 4 and 5 as well as the ODEParameters and Static Parametes. Our cosntraint
is written assuming all of the variable from 1 phase are grouped together and that in these subgroups, state varibales
come before ode parameter variables, and static parameter variables. We specify the specific state,odeparam, and static param 
variable indices for each phase using the indexing vectors,xtuvars,opvars, and spvars respectively.
'''

def ContCon():
    p4x6,p4o0,p4s0,p5x6,p5o0,p5s0 = Args(6).tolist()
   
    f= vf.stack([p4x6-p5x6,
                 p4o0-p5o0,
                 p4s0-p5s0])
    
    return f

phase_regs     = ["Back","Front"]
phases_to_link = [[4 , 5]]
xtuvars        = [[6],[6]]
opvars         = [[0],[0]]
spvars         = [[0],[0]]
lpvars =         []

ocp.addLinkEqualCon(ContCon(),phase_regs,phases_to_link,xtuvars,opvars,spvars,lpvars)

'''
Finally lets examine how we can incorporate 


'''



