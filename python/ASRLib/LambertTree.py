# -*- coding: utf-8 -*-
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

def LambertTreeSearchImpl(mu,Tables,T0s,LegTOFs,integ,ConcatLegs=False):
        
        
        Lens =[len(T0s)] +[len(LegT) for LegT in LegTOFs]
        Idxs = np.indices(Lens).T.reshape(-1,len(Lens))
        
        Tseqs=[]
        
        for Id in Idxs:
            ts = [T0s[Id[0]]]
            for i in range(1,len(Id)):
                ts.append(ts[-1] + LegTOFs[i-1][Id[i]])
            Tseqs.append(ts)
        
        def LegDV(Tab1,t1,Tab2,t2,lw,Integ=False):
            
            tof = t2-t1
            
            X1 =  Tab1.Interpolate(t1)
            X2 =  Tab2.Interpolate(t2)
            
            V1,V2 = ast.Astro.lambert_izzo(X1[0:3], X2[0:3],tof,mu,lw)
            DV = np.linalg.norm(V1-X1[3:6]) + np.linalg.norm(V2-X2[3:6])
            if(Integ==False):
                return DV
            else:
                IGS = np.copy(X1)
                IGS[3:6] = V1
                Traj = integ.integrate_dense(IGS,t2)
                return Traj
       
        BestTraj=[]
        DVs=[]
        SLseqs=[]
        for i,Seq in enumerate(Tseqs):
            print(float(i)/float(len(Tseqs)))
            DV = 0
            sl =[]
            for i in range(0,len(Seq)-1):
                Tab1 = Tables[i]
                Tab2 = Tables[i+1]
                t1  = Seq[i]
                t2  = Seq[i+1]
                DVL = LegDV(Tab1,t1,Tab2,t2,True)
                DVS = LegDV(Tab1,t1,Tab2,t2,False)
                flip = bool(random.randint(0, 1))

                if(flip==False):
                    DV+=DVS
                    sl.append(False)
                else:
                    DV+=DVL
                    sl.append(True)
                
            DVs.append(DV)
            SLseqs.append(sl)
            
        idx = np.argmin(DVs)
        Seq = Tseqs[idx]
        sl  = SLseqs[idx]
        BestTraj=[]
        for i in range(0,len(Seq)-1):
            Tab1 = Tables[i]
            Tab2 = Tables[i+1]
            t1  = Seq[i]
            t2  = Seq[i+1]
            BestTraj.append(LegDV(Tab1,t1,Tab2,t2,sl[i],integ))

        return BestTraj



def solverp(rp, mu, Vvec):
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

def vp(rp, mu, V):
    return abs(np.sqrt(np.linalg.norm(V[0]) + 2.0*mu/rp) - np.sqrt(np.linalg.norm(V[1]) + 2.0*mu/rp))

def GA(TOFs, Tables, integ, mu, numseq, MuRad, keys):
    #Tables = args[0]
    #integ = args[1]
    #mu = args[2]
    def LegDV(Tab1,t1,Tab2,t2,lw,Integ=False):
        
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
            Traj = integ.integrate_dense(IGS,t2)
            return Traj, np.array(DVvec)
        
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
    for i in range(0,len(Seq)-1):
        Tab1 = Tables[i]
        Tab2 = Tables[i+1]
        t1  = Seq[i]
        t2  = Seq[i+1]
        DVL, DVvecL = LegDV(Tab1,t1,Tab2,t2,True)
        DVS, DVvecS = LegDV(Tab1,t1,Tab2,t2,False)
        DVVL.append(DVvecL)
        DVVS.append(DVvecS)
        flip = flipvec[i]
        '''
        if(flip==False):
           DV+=DVS
        else:
            DV+=DVL
        '''
    
    DV  = 0
    FullDV = []
    flip = flipvec[0]
    V = 0
    if flip ==False:
        V = np.linalg.norm(DVVS[0][0][1] - DVVS[0][0][0])
    else:
        V = np.linalg.norm(DVVL[0][0][1] - DVVL[0][0][0])
    DV += V
    
    
    flip = flipvec[-1]
    V = 0
    if flip ==False:
        V = np.linalg.norm(DVVS[-1][0][3] - DVVS[-1][0][2])
    else:
        V = np.linalg.norm(DVVL[-1][0][3] - DVVL[-1][0][2])
    DV += V

    for i in range(1, len(DVVL)):
        flip = flipvec[i]
        if flip ==False:
            #inbound rel to body
            if flipvec[i-1] == False:
                V1 = DVVS[i][0][1] - DVVS[i-1][0][2]
                #outbound
                V2 = DVVS[i][0][1] - DVVS[i][0][0]
                Vinout = [V1, V2]
                rp = solverp(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = vp(rp, MuRad[keys[i]][0],Vinout)

                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0
                DV+=V
            else:
                V1 = DVVS[i][0][1] - DVVL[i-1][0][2]
                #outbound
                V2 = DVVS[i][0][1] - DVVS[i][0][0]
                Vinout = [V1, V2]
                rp = solverp(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = vp(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0   

                DV+=V
                
        else:
            if flipvec[i-1] == False:
                V1 = DVVL[i][0][1] - DVVS[i-1][0][2]
                #outbound
                V2 = DVVL[i][0][1] - DVVL[i][0][0]
                Vinout = [V1, V2]
                rp = solverp(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = vp(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0

                DV+=V
            else:
                #inbound rel to body
                V1 = DVVL[i][0][1] - DVVL[i-1][0][2]
                #outbound
                V2 = DVVL[i][0][1] - DVVL[i][0][0]
                Vinout = [V1, V2]
                rp = solverp(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
            
                V = vp(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0
 
                DV+=V
   
    return DV

def LambertSeq(TOFs, Tables, sl, numseq, integ, mu, MuRad, keys):
  
    def LegDV(Tab1,t1,Tab2,t2,lw,Integ=False):
        
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
            Traj = integ.integrate_dense(IGS,t2)
            return Traj, np.array(DVvec)
    Seq = [TOFs[0]]
    for i in range(1, len(TOFs[:numseq])):
        Seq.append(Seq[i - 1] + TOFs[i])
    Seq = np.array(Seq)
    BestTraj=[]
    flipvec = sl
    DVVL = []
    DVVS = []
    for i in range(0,len(Seq)-1):
            Tab1 = Tables[i]
            Tab2 = Tables[i+1]
            t1  = Seq[i]
            t2  = Seq[i+1]
            DVL, DVvecL = LegDV(Tab1,t1,Tab2,t2,True)
            DVS, DVvecS = LegDV(Tab1,t1,Tab2,t2,False)
            DVVL.append(DVvecL)
            DVVS.append(DVvecS)
            BestTraj.append(LegDV(Tab1,t1,Tab2,t2,sl[i],integ)[0])
            
    #Departure   
    DV  = 0
    FullDV = []
    flip = flipvec[0]
    if flip ==False:
        V = np.linalg.norm(DVVS[0][0][1] - DVVS[0][0][0])
    else:
        V = np.linalg.norm(DVVL[0][0][1] - DVVL[0][0][0])
    DV += V
    FullDV.append(V)
    
    for i in range(1, len(DVVL)):
        flip = flipvec[i]
        if flip ==False:
            #inbound rel to body
            if flipvec[i-1] == False:
                V1 = DVVS[i][0][1] - DVVS[i-1][0][2]
                #outbound
                V2 = DVVS[i][0][1] - DVVS[i][0][0]
                Vinout = [V1, V2]
                rp = solverp(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = vp(rp, MuRad[keys[i]][0],Vinout)

                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0
                FullDV.append(V)
                DV+=V
            else:
                V1 = DVVS[i][0][1] - DVVL[i-1][0][2]
                #outbound
                V2 = DVVS[i][0][1] - DVVS[i][0][0]
                Vinout = [V1, V2]
                rp = solverp(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = vp(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0   
                FullDV.append(V)
                DV+=V
                
        else:
            if flipvec[i-1] == False:
                V1 = DVVL[i][0][1] - DVVS[i-1][0][2]
                #outbound
                V2 = DVVL[i][0][1] - DVVL[i][0][0]
                Vinout = [V1, V2]
                rp = solverp(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
                V = vp(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0
                FullDV.append(V)
                DV+=V
            else:
                #inbound rel to body
                V1 = DVVL[i][0][1] - DVVL[i-1][0][2]
                #outbound
                V2 = DVVL[i][0][1] - DVVL[i][0][0]
                Vinout = [V1, V2]
                rp = solverp(MuRad[keys[i]][0], MuRad[keys[i]][1], Vinout)
            
                V = vp(rp, MuRad[keys[i]][0],Vinout)
                if np.isnan(V):
                    V = 100000.0
                elif rp < MuRad[keys[i]][0]:
                    V = 100000.0
                FullDV.append(V)
                DV+=V
            
    #Arrival
    flip = flipvec[-1]
    if flip ==False:
        V = np.linalg.norm(DVVS[-1][0][3] - DVVS[-1][0][2])
    else:
        V = np.linalg.norm(DVVL[-1][0][3] - DVVL[-1][0][2])
    DV += V
    FullDV.append(V)
    
    return BestTraj, FullDV




        

