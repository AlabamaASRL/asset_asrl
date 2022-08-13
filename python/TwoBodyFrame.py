import numpy as np
import asset as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

class TwoBodyFrame():
    def __init__(self,P1mu,lstar):
        self.mu = 1
        self.P1mu   = P1mu
        self.lstar  = lstar
        self.tstar  = np.sqrt((lstar**3)/(P1mu))
        self.vstar  = lstar/self.tstar
        self.astar  = (P1mu)/(lstar**2)
        self.mustar = (P1mu)
    def TwoBodyEOMs(self,r,v,otherAccs=[],otherEOMs=[]):
        accG = (-self.mu )* r.normalized_power3() 
        accT = vf.Sum([accG ]+ otherAccs)
        return vf.Stack([v, accT]+otherEOMs)