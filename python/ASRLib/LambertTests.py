from AstroModels import NBodyFrame,NBody,NBody_LT
from AstroConstraints import RendezvousConstraint,CosAlpha
from FramePlot import CRPlot,TBPlot,plt
from LambertGA import LambertGASeq
import MKgSecConstants as c
import numpy as np
import asset as ast
import Date as DT
from MiscUtils import grid_to_points
import scipy as scp

def normalized(x): return x/np.linalg.norm(x)


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

JD0 = DT.date_to_jd(2024, 9, 15)
JDF = DT.date_to_jd(2040, 9, 15)

NFrame = NBodyFrame("SUN",c.MuSun,c.AU,JD0,JDF,SpiceFrame='ECLIPJ2000')


ETab = NFrame.GetSpiceBodyTable("EARTH", 8000)
MTab = NFrame.GetSpiceBodyTable("MARS", 6000)
PTab = NFrame.GetSpiceBodyTable("Psyche", 6000)
JTab = NFrame.GetSpiceBodyTable("Jupiter Barycenter", 6000)
UTab = NFrame.GetSpiceBodyTable("Saturn Barycenter", 6000)
PH27Tab = NFrame.GetSpiceBodyTable("54186922", 8000)
VTab = NFrame.GetSpiceBodyTable("Venus", 6000)

ode = NBody(NFrame,Enable_P1_Acc=False)
integ=ode.integrator(.01)
integ.Adaptive = True

plot = TBPlot(ode,"SUN")




T0s    = np.linspace(1.0,20.0,30)
MTOFs  = np.linspace(0.9,13.0,20)
JTOFs  = np.linspace(10.0,80.0,20)
UTOFs  = np.linspace(25.0,60.0,45)
PTOFs  = np.linspace(6.0,18.0,20)


#Trajs= LambertTreeSearchImpl(1.0, [ETab,VTab,MTab, JTab],T0s,[MTOFs,MTOFs,JTOFs],integ)
'''
bounds = [(1.0, 60.),  (.9, 18.), (7., 40.), (0, np.pi), (0, np.pi),  (0, np.pi)]
TOF = [5.0,  5.0, 30.0, np.pi/2, np.pi/2,  np.pi/2]
Tables = [ETab,MTab, PTab]
numseq = len(Tables)
ga = scp.optimize.differential_evolution(GA, bounds, args = (Tables,integ, 1.0, len(Tables)), popsize = 40, mutation=(.5, 1.7))
TOFs = ga.x[:numseq]
sl = []
for i,val in enumerate(ga.x[numseq:]):
    if np.cos(val) > 0:
        sl.append(True)
    else:
        sl.append(False)

Trajs = LambertSeq(TOFs, Tables, sl, numseq, integ, 1.0)
'''
SEQ = [["EARTH","Venus", "EARTH", "Jupiter Barycenter"]]
iters = 400
TT, DVs = LambertGASeq(SEQ, iters, strat = 'best1bin')


Trajs1 = TT[0]



plot.addTraj(ETab.InterpNonDim(1000,0.01,.99), "Earth","g")
#plot.addTraj(MTab.InterpNonDim(3000,0.01,.99), "Mars","r")
plot.addTraj(JTab.InterpNonDim(1000,0.01,.99), "Jupiter","k")
plot.addTraj(VTab.InterpNonDim(3000,0.01,.99), "Venus","orange")
plot.addTraj(PH27Tab.InterpNonDim(3000,0.01,.99), "PH27","k")
#plot.addTraj(UTab.InterpNonDim(1000,0.01,.99), "Uranus","cyan")
#plot.addTraj(PTab.InterpNonDim(1000,0.01,.99), "Psyche","blue")

plot.addTrajSeq(Trajs1,"M")

plot.Plot2d()
plot.Plot3d(bbox="One")

print(NFrame.JD_to_NDTime((JD0+JDF)/2.0))