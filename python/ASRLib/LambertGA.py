import numpy as np
import asset as ast
import random
import Date as DT
import scipy as scp
import MKgSecConstants as c
from AstroModels import NBodyFrame,NBody,NBody_LT
from scipy.optimize import NonlinearConstraint
from LambertGAHelper import PeriDV, FindPeriRadius, LegDV, ComputeDVSeq, LambertSeq
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

def GA(TOFs, Tables, integ, mu, numseq, MuRad, keys):
    Seq = [TOFs[0]]
    for i in range(1, len(TOFs[:numseq])):
        Seq.append(Seq[i - 1] + TOFs[i])
    Seq = np.array(Seq)
    DV = 0
    flipvec = []
    for i, val in enumerate(TOFs[numseq:]):
        if np.cos(val) > 0:
            flipvec.append(True)
        elif np.cos(val) < 0:
            flipvec.append(False)
    DVVL = []
    DVVS = []
    #Evaluate Short and Long Way
    for i in range(0,len(Seq)-1):
        Tab1 = Tables[i]
        Tab2 = Tables[i+1]
        t1  = Seq[i]
        t2  = Seq[i+1]
        DVL, DVvecL = LegDV(Tab1,t1,Tab2,t2,True, mu)
        DVS, DVvecS = LegDV(Tab1,t1,Tab2,t2,False, mu)
        DVVL.append(DVvecL)
        DVVS.append(DVvecS)
        flip = flipvec[i]

    DV = ComputeDVSeq(MuRad, flipvec, DVVL, DVVS, keys)
    return DV



def LambertGASeq(SeqsVec, iters, strat = 'best1bin'):
    JD0 = DT.date_to_jd(2021, 9, 15)
    JDF = DT.date_to_jd(2049, 9, 15)
    
    NFrame = NBodyFrame("SUN",c.MuSun,c.AU,JD0,JDF,SpiceFrame='ECLIPJ2000')
    
    ode = NBody(NFrame,Enable_P1_Acc=False)
    integ=ode.integrator(.01)
    integ.Adaptive = True
    TT = []
    DVs = []
    for i, vec in enumerate(SeqsVec):
        Ubodies = np.unique(vec)
        bodydata = []
        bodytab = {}
        for body in Ubodies:
            bodydata = NFrame.GetSpiceBodyTable(body, 8000)
            bodytab[body] = bodydata
        numseq = len(vec)
        bounds = []
        TOF = []
        for j in range(numseq):
            bounds.append((.7, 50.))
            TOF.append(5.0)
        for j in range(numseq):
            bounds.append((0, np.pi))
            TOF.append(np.pi/2.)
        Tables = []
        for j in range(len(vec)):
            Tables.append(bodytab[vec[j]])
            
            
        lstar     = c.AU
        vstar     = np.sqrt(c.MuSun/lstar)
        tstar     = np.sqrt(lstar**3/c.MuSun)
        Stuff ={}
        Stuff["EARTH"] = [(c.RadiusEarth + 300.0*1000.0)/lstar, c.MuEarth*(tstar**2)/(lstar**3)]
        Stuff["Venus"] = [(c.RadiusVenus + 300.0*1000.0)/lstar, c.MuVenus*(tstar**2)/(lstar**3)]
        Stuff["Jupiter Barycenter"] = [(4092938844.5312696)/lstar, c.MuJupiter*(tstar**2)/(lstar**3)]
        
        mu = 1.0
        ga = scp.optimize.differential_evolution(GA, bounds, args = (Tables,integ, 1.0, len(Tables), Stuff, vec),strategy = strat, popsize = 200, mutation=(.5, 1.7), disp= True, maxiter = iters)
        DVs.append(ga.fun)
        TOFs = ga.x[:numseq]
        sl = []
        for i,val in enumerate(ga.x[numseq:]):
            if np.cos(val) > 0:
                sl.append(True)
            else:
                sl.append(False)
        Trajs, FullDV = LambertSeq(TOFs, Tables, integ, mu, numseq, Stuff, vec, sl)
        TT.append(Trajs)
        print(FullDV)
    return TT, DVs