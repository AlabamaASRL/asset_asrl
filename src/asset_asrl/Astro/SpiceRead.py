import numpy as np
import spiceypy as sp
import asset_asrl.Astro.Constants as c
import asset_asrl.Astro.Date as dt
#sp.furnsh('BasicKernel.txt')

def GetEphemTraj2(body,startJD,endJD,numstep,LU=1.0,TU=1.0,Frame='ECLIPJ2000',Center='SOLAR SYSTEM BARYCENTER'):
    times = [dt.JD_SPJ2000D(float(x)*(endJD-startJD)/float(numstep) + startJD) for x in range(numstep)]
    states=[]
    t0 = times[0]
    for t in times:
        X = np.zeros((7))
        X[0:6] = sp.spkezr(body,t,Frame,'NONE',Center)[0]
        X[0:3] *= 1000.0/LU
        X[3:6] *= 1000.0*TU/LU
        X[6] = (t-t0)/TU
        states.append(X)
    return states

def PoleVector(IAUBodyFrame, TargetFrame, startJD,endJD,numstep,TU=1.0):
    times = [dt.JD_SPJ2000D(float(x)*(endJD-startJD)/float(numstep) + startJD) for x in range(numstep)]
    poles = []
    t0 = times[0]
    for t in times:
        X = np.zeros((4))
        
        Mat = sp.pxform(IAUBodyFrame,TargetFrame,t)
        X[0] = Mat[0,2]
        X[1] = Mat[1,2]
        X[2] = Mat[2,2]
        X[3] = (t-t0)/TU
        poles.append(X)
    return poles
        
def SpiceFrameTransform(FromFrame,ToFrame,X,JD):
    et =dt.JD_SPJ2000D(JD)
    Mat = sp.sxform(FromFrame,ToFrame,et)
    return np.dot(Mat,X)

def GetEphemTraj(body,startJD,endJD,numstep,LU=1.0,TU=1.0,\
    Frame='ECLIPJ2000',Center='SOLAR SYSTEM BARYCENTER', useET = False):

    # Option for use of ET as input to function.
    if not useET:
        times = [dt.JD_SPJ2000D(x*(endJD-startJD)/numstep + startJD) for x in range(numstep)]
    else:
        times = [startJD + (endJD - startJD)*x/numstep for x in range(numstep)]

    states=[]
    t0 = times[0]
    for t in times:
        X = np.zeros((7))
        X[0:6] = sp.spkezr(body,t,Frame,'NONE',Center)[0]
        X[0:3] *= 1000.0/LU
        X[3:6] *= 1000.0*TU/LU
        X[6] = (t-t0)/TU
        states.append(X)
    return states


def GetEphemState(body,JD,LU=1.0,TU=1.0,Frame='ECLIPJ2000',Center='SOLAR SYSTEM BARYCENTER'):
    t=dt.JD_SPJ2000D(JD)
    X = np.zeros((6))
    X[0:6] = sp.spkezr(body,t,Frame,'NONE',Center)[0]
    X[0:3] *= 1000.0/LU
    X[3:6] *= 1000.0*TU/LU
    return X
