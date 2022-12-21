import asset as ast
import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import default_rng
import scipy.io
import time
################################################################################
# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
tModes = oc.TranscriptionModes
