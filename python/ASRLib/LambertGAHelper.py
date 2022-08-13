import numpy as np
import asset as ast
import random
import scipy as scp
import Date as DT
import MKgSecConstants as c
from AstroModels import NBodyFrame,NBody,NBody_LT
from scipy.optimize import NonlinearConstraint
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags

def FindPeriRadius(rp, mu, Vvec):
    turnangle = np.arccos(np.dot(Vvec[0], Vvec[1])/(np.linalg.norm(Vvec[0]) * np.linalg.norm(Vvec[1])))
    #turnangle = np.dot(Vvec[0], Vvec[1])
    rp = rp
    rpold = 0.
    tol = 1e-7
    vinfinc = np.linalg.norm(Vvec[0])
    vinfout = np.linalg.norm(Vvec[1])

    count = 0
    while abs(rpold - rp) > tol and count < 50:
        firstterm = -(mu*vinfinc)/(((vinfinc*rp+mu)**2)*np.sqrt(1.0-mu**2/(vinfinc*rp+mu)**2))
        secondterm = -(mu*vinfout)/(((vinfout*rp+mu)**2)*np.sqrt(1.0-mu**2/(vinfout*rp+mu)**2))
        total = firstterm + secondterm
        eqn = (np.arcsin(mu/(mu + rp*vinfinc)) + np.arcsin(mu/(mu + rp*vinfout))) - turnangle
        rpold = rp
        rp = rp - (eqn/total)
        #rp = abs(rp)
        count+=1
    return rp

def PeriDV(rp, mu, V):
    return abs(np.sqrt(np.linalg.norm(V[0]) + 2.0*mu/rp) - np.sqrt(np.linalg.norm(V[1]) + 2.0*mu/rp))

def LegDV(Tab1,t1,Tab2,t2,lw, mu, Integ=False):
        tof = t2-t1
        DVvec = []
        X1 =  Tab1.Interpolate(t1)
        X2 =  Tab2.Interpolate(t2)
        V1,V2 = ast.Astro.lambert_izzo(X1[0:3], X2[0:3],tof,mu,lw)
        DV = np.linalg.norm(V1-X1[3:6]) + np.linalg.norm(V2-X2[3:6]) 
        DVvec.append([V1, X1[3:6], V2, X2[3:6]])
        if(Integ==False):
            return DV, np.array(DVvec)
        else:
            IGS = np.copy(X1)
            IGS[3:6] = V1
            Traj = Integ.integrate_dense(IGS,t2)
            return Traj, np.array(DVvec)
        
        
def ComputeDVSeq(MuRad, FlipVec, DVVL, DVVS, keys, listflag = False):
    fullDV = []
    DV  = 0
    flip = FlipVec[0]
    V = 0
    if flip ==False:
        V = np.linalg.norm(DVVS[0][0][1] - DVVS[0][0][0])
    else:
        V = np.linalg.norm(DVVL[0][0][1] - DVVL[0][0][0])
        
    dv = np.sqrt(V**2 + 2.0*MuRad[keys[0]][1]/MuRad[keys[0]][0]) - np.sqrt(MuRad[keys[0]][1]/MuRad[keys[0]][0])
    DV += dv
    fullDV.append(dv)
    
    
    flip = FlipVec[-1]
    V = 0
    if flip ==False:
        V = np.linalg.norm(DVVS[-1][0][3] - DVVS[-1][0][2])
    else:
        V = np.linalg.norm(DVVL[-1][0][3] - DVVL[-1][0][2])

    e = .99
    rp = MuRad[keys[-1]][0] * (1.0 - e)
    ra = MuRad[keys[-1]][0] * (1.0 + e)
    dv = np.sqrt(V**2 + 2.0*MuRad[keys[-1]][1]/rp) - np.sqrt((2.0*MuRad[keys[-1]][1] *ra)/(rp * (ra + rp)) )
    fullDV.append(dv)
    DV += dv
    
    for i in range(1, len(DVVL)):
        flip = FlipVec[i]
        if flip ==False:
            #inbound rel to body
            if FlipVec[i-1] == False:
                V1 = DVVS[i][0][1] - DVVS[i-1][0][2]
                #outbound
                V2 = DVVS[i][0][1] - DVVS[i][0][0]
                Vinout = [V1, V2]
                rp = FindPeriRadius(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = PeriDV(rp, MuRad[keys[i]][0],Vinout)

                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0
                fullDV.append(V)
                DV+=V
            else:
                V1 = DVVS[i][0][1] - DVVL[i-1][0][2]
                #outbound
                V2 = DVVS[i][0][1] - DVVS[i][0][0]
                Vinout = [V1, V2]
                rp = FindPeriRadius(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = PeriDV(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0 
                fullDV.append(V)
                DV+=V
                
        else:
            if FlipVec[i-1] == False:
                V1 = DVVL[i][0][1] - DVVS[i-1][0][2]
                #outbound
                V2 = DVVL[i][0][1] - DVVL[i][0][0]
                Vinout = [V1, V2]
                rp = FindPeriRadius(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = PeriDV(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0
                fullDV.append(V)
                DV+=V
            else:
                #inbound rel to body
                V1 = DVVL[i][0][1] - DVVL[i-1][0][2]
                #outbound
                V2 = DVVL[i][0][1] - DVVL[i][0][0]
                Vinout = [V1, V2]
                rp = FindPeriRadius(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
            
                V = PeriDV(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0
                fullDV.append(V)
                DV+=V
    if listflag == False:
        return DV
    else:
        return DV, fullDV
    
def LambertSeq(TOFs, Tables, integ, mu, numseq, MuRad, keys, flipvec):
    Seq = [TOFs[0]]
    for i in range(1, len(TOFs)):
        Seq.append(Seq[i - 1] + TOFs[i])
    Seq = np.array(Seq)
    Traj = []
    #Evaluate Short and Long Way
    DVVL = []
    DVVS = []
    #Evaluate Short and Long Way
    for i in range(0,len(Seq)-1):
        Tab1 = Tables[i]
        Tab2 = Tables[i+1]
        t1  = Seq[i]
        t2  = Seq[i+1]
        DVL, DVvecL = LegDV(Tab1,t1,Tab2,t2,mu, True)
        DVS, DVvecS = LegDV(Tab1,t1,Tab2,t2,mu, False)
        DVVL.append(DVvecL)
        DVVS.append(DVvecS)
        Traj.append(LegDV(Tab1,t1,
                          Tab2,t2,mu,flipvec[i], integ)[0])
        
    DV = ComputeDVSeq(MuRad, flipvec, DVVL, DVVS, keys, True)[1]
    
    return Traj, DV
    
    