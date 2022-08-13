import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from numpy import cos,sin,tan,arccos,arcsin,arctan2
from scipy.spatial.transform import Rotation as R
import MKgSecConstants as c


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes

