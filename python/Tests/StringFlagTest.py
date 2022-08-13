import numpy as np
import asset_asrl as ast


oc = ast.OptimalControl

ocp = ast.OptimalControl.OptimalControlProblem()


##############################################

ocp.optimizer.set_OptLSMode("AUGLANG")
print(ocp.optimizer.OptLSMode)

ocp.optimizer.set_OptLSMode("LANG")
print(ocp.optimizer.OptLSMode)

ocp.optimizer.set_OptLSMode("NOLS")
print(ocp.optimizer.OptLSMode)

ocp.optimizer.set_OptLSMode("L1")
print(ocp.optimizer.OptLSMode)


###############################################

ocp.optimizer.set_OptBarMode("PROBE")
print(ocp.optimizer.OptBarMode)

ocp.optimizer.set_OptBarMode("LOQO")
print(ocp.optimizer.OptBarMode)


ocp.optimizer.set_SoeBarMode("PROBE")
print(ocp.optimizer.SoeBarMode)

ocp.optimizer.set_SoeBarMode("LOQO")
print(ocp.optimizer.SoeBarMode)
#################################################

ocp.optimizer.set_QPOrderingMode("PARMETIS")
print(ocp.optimizer.QPOrderingMode)

ocp.optimizer.set_QPOrderingMode("METIS")
print(ocp.optimizer.QPOrderingMode)

ocp.optimizer.set_QPOrderingMode("MINDEG")
print(ocp.optimizer.QPOrderingMode)

##################################################

print(oc.strto_PhaseRegionFlag("Front"))
print(oc.strto_PhaseRegionFlag("First"))
print(oc.strto_PhaseRegionFlag("Back"))
print(oc.strto_PhaseRegionFlag("Last"))
print(oc.strto_PhaseRegionFlag("Path"))
