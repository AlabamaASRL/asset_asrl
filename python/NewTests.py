import numpy as np
import asset as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags


args = Args(8)
x = [1,2,3,1,1,1,1,1]


f1 = vf.cwiseQuotient(args.head3(),
                      args.tail3())

f2 = args.head3()*(7*args[4])

f3 = vf.exp(args[1])*args[0]*(args[3])

f1.rpt(x,1000000)
f2.rpt(x,1000000)
f3.rpt(x,1000000)
