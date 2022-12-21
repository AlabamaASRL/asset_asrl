from   asset.OptimalControl import *
import asset as _asset
import inspect


###############################################################################
## Exposing Compiled Module Elements to improve autocomplete

ControlModes = _asset.OptimalControl.ControlModes
FiniteDiffTable = _asset.OptimalControl.FiniteDiffTable
IntegralModes = _asset.OptimalControl.IntegralModes
InterpFunction = _asset.OptimalControl.InterpFunction
TranscriptionModes = _asset.OptimalControl.TranscriptionModes

InterpFunction_1 = _asset.OptimalControl.InterpFunction_1
InterpFunction_3 = _asset.OptimalControl.InterpFunction_3
InterpFunction_6 = _asset.OptimalControl.InterpFunction_6
LGLInterpTable = _asset.OptimalControl.LGLInterpTable

LinkConstraint = _asset.OptimalControl.LinkConstraint
LinkFlags = _asset.OptimalControl.LinkFlags
LinkObjective = _asset.OptimalControl.LinkObjective

ODEArguments = _asset.OptimalControl.ODEArguments
OptimalControlProblem = _asset.OptimalControl.OptimalControlProblem
PhaseInterface = _asset.OptimalControl.PhaseInterface
PhaseRegionFlags = _asset.OptimalControl.PhaseRegionFlags
RKOptions = _asset.OptimalControl.RKOptions
StateConstraint = _asset.OptimalControl.StateConstraint
StateObjective = _asset.OptimalControl.StateObjective

ode_2_1 = _asset.OptimalControl.ode_2_1
ode_6 = _asset.OptimalControl.ode_6
ode_6_3 = _asset.OptimalControl.ode_6_3
ode_7_3 = _asset.OptimalControl.ode_7_3
ode_x = _asset.OptimalControl.ode_x
ode_x_u = _asset.OptimalControl.ode_x_u
ode_x_u_p = _asset.OptimalControl.ode_x_u_p

###############################################################################
## Expose Pure Python Extensions to OptimalControl

from .ODEBaseClass import ODEBase



##############################################################################
if __name__ == "__main__":
    
    mlist = inspect.getmembers(_asset.OptimalControl)
    for m in mlist:print(m[0],'= _asset.OptimalControl.'+str(m[0]))
    
    
    


