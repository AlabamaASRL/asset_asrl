# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:07:15 2022

@author: gerde
"""

from AstroModels import EPPRFrame,EPPR,EPPR_SolarSail,SolarSail,EPPR_LT,LowThrustAcc
from AstroConstraints import RendezvousConstraint,CosAlpha
from FramePlot import CRPlot,TBPlot,plt
import MKgSecConstants as c
import numpy as np
import asset as ast
from DerivChecker import FDDerivChecker
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

vf = ast.VectorFunctions
oc = ast.OptimalControl


ast.SoftwareInfo()

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
LinkRegs = oc.LinkFlags
sol = ast.Solvers

######## Initial Conditions ###################
IS1 = np.array([-7.666349e+03, -4.222609e+03, -3.404840e+03, 2.023496e-01, -8.166659e+00, -4.179644e+00])
IS1 = IS1*1000.

IS2 = np.array([-6.790120e+03, -5.329943e+03, -3.695847e+03, 1.596999e+00, -8.090864e+00,-4.022860e+00])
IS2 = IS2*1000.

IS3 = np.array([-5.193037e+03, -6.670989e+03, -4.094500e+03, 3.578327e+00,-7.625257e+00, -3.646662e+00])
IS3 = IS3*1000.

ISnom = np.array([-7.660922645448107e+03, -4.459416484559431e+03, -3.526108320344569e+03, 3.066786833683193e-01, -8.106509619387344e+00, -4.131927337896593e+00])
ISnom = ISnom*1000.
JDnom = 2460707.56869

JD1 = 2460707.53432
JD2 = 2460717.51632
JD3 = 2460727.49832
#############################################

ISs = [ISnom, IS1, IS2, IS3]
JDs = [JDnom, JD1, JD2, JD3]
colors = ['k', 'r', 'b', 'orange']

JDImap = 2460585.02936299
JD0 = JDImap - 3.5
JDF = JD0 + 3.0*365.0   
N = 4000

SpiceFrame = 'J2000'
EFrame = EPPRFrame("SUN",c.MuSun,"EARTH",c.MuEarth,c.AU,JD0,JDF,N=N,SpiceFrame=SpiceFrame)
Bodies = ["MOON","JUPITER BARYCENTER","VENUS"]
EFrame.AddSpiceBodies(Bodies,N=2000)
EFrame.Add_P2_J2Effect()


eppr  = EPPR(EFrame,Enable_J2=True)

beta = 0.02
SailModel = SolarSail(beta,False)
sail = EPPR_SolarSail(EFrame,SailModel = SailModel)


epinteg =eppr.integrator(c.pi/50000)
epinteg.Adaptive=True

sinteg =sail.integrator(c.pi/7000,(Args(3)).normalized(),range(0,3))
sinteg.Adaptive=True

Day = c.day/EFrame.tstar
plot= CRPlot(EFrame)

for i, ig in enumerate(ISs):
    Imap  = EFrame.Vector_to_Frame(ig, JDs[i],center="P2")
    IG = Imap[0]
    Tdep = 116.5*Day
    Ts = 197*Day
    ImapNom = epinteg.integrate_dense(IG,IG[6] + Tdep)
    IT = epinteg.integrate_dense(IG,IG[6] + 190*Day,100000)
    ITab = oc.LGLInterpTable(6,IT,len(IT)*4)
    
    IG = np.zeros((10))
    IG[0:7]=ImapNom[-1]
    IG[7]=1.0
    plot.addTraj(ImapNom,"IMAP-Nominal"+str(i),colors[i])
MoonTraj = EFrame.GetSpiceBodyEPPRTraj("MOON", 4000)


plot.addTraj(MoonTraj[3000:], 'Moon', 'grey')
plot.Plot3d(pois=['L1','P2','L2'],bbox='L1P2',legend=True)


