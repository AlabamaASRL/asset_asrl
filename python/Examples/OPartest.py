import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from DerivChecker import FDDerivChecker
import sys
import jedi

ast.PyMain()
vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes

ff = Args(2)

PhaseRegs = oc.PhaseRegionFlags

X,ang = Args(7).tolist([(0,6),(6,1)])


rowlist = vf.stack(vf.cos(ang), -vf.sin(ang), 0, 0, 0, 0,
               vf.sin(ang), vf.cos(ang), 0, 0, 0, 0,
               0, 0, 1, 0, 0, 0,
               -vf.sin(ang), -vf.cos(ang), 0, vf.cos(ang), -vf.sin(ang), 0,
               vf.cos(ang), -vf.sin(ang), 0, vf.sin(ang), vf.cos(ang), 0,
               0, 0, 0, 0, 0, 1)
               


M = vf.RowMatrix(rowlist,6,6)

F = M*X.normalized()

Xin = np.ones((7))

FDDerivChecker(F,Xin)

